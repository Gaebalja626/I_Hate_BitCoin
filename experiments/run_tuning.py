"""2024~2025 구간에서 lookahead x dead_zone 그리드 서치 튜닝"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

os.environ["SSL_CERT_FILE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"
os.environ["REQUESTS_CA_BUNDLE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"

import config as cfg
from data_collector import load_or_fetch
from models import lgbm_model
from backtester import run_3action
from feature_engineer import create_features_v2
from experiment import create_experiment, save_metrics

raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, cfg.DATA_END, cfg.TECH_INDICATORS)

configs = []
for la in [6, 12, 24, 48]:
    for dz in [0.003, 0.005, 0.007, 0.01, 0.015, 0.02]:
        configs.append((la, dz))

header = f"{'LA':>4} | {'DZ':>6} | {'Iter':>5} | {'Buy':>5} | {'Hold':>5} | {'Sell':>5} | {'Return':>8} | {'B&H':>8} | {'MDD':>8} | {'Sharpe':>7} | {'Trades':>6}"
print(header)
print("-" * len(header))

best_ret = -999
best_sharpe = -999
best_cfg = None
all_results = []

for la, dz in configs:
    df, fcols = create_features_v2(raw, cfg.TECH_INDICATORS, lookahead=la, buy_threshold=dz)
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[df["date"] < cfg.TEST_START].reset_index(drop=True)
    test_df = df[df["date"] >= cfg.TEST_START].reset_index(drop=True)

    if len(test_df) < 100:
        continue

    model, feat_imp, info = lgbm_model.train_3class(train_df, fcols)
    actions, probs = lgbm_model.predict_3class(model, test_df, fcols)
    result = run_3action(actions, test_df)

    nb = int(sum(actions == 0))
    nh = int(sum(actions == 1))
    ns = int(sum(actions == 2))
    ret = result["model_return_pct"]
    sharpe = result["sharpe_ratio"]

    row = {
        "lookahead": la, "dead_zone": dz,
        "best_iter": info["best_iteration"],
        "buy": nb, "hold": nh, "sell": ns,
        **result,
    }
    all_results.append(row)

    if ret > best_ret:
        best_ret = ret
        best_cfg = (la, dz)
        best_model = model
        best_feat_imp = feat_imp
        best_result = result
        best_actions = actions
        best_test_df = test_df
        best_info = info

    print(
        f"{la:>3}h | {dz*100:>5.1f}% | {info['best_iteration']:>5} | "
        f"{nb:>5} | {nh:>5} | {ns:>5} | "
        f"{ret:>+7.2f}% | {result['buyhold_return_pct']:>+7.2f}% | "
        f"{result['mdd_pct']:>7.2f}% | {result['sharpe_ratio']:>6.2f} | "
        f"{result['total_trades']:>6}"
    )

print(f"\n{'='*60}")
print(f"[Best] lookahead={best_cfg[0]}h, dead_zone={best_cfg[1]*100:.1f}%")
print(f"  Return: {best_ret:+.2f}%")
print(f"  B&H: {best_result['buyhold_return_pct']:+.2f}%")
print(f"  MDD: {best_result['mdd_pct']:.2f}%")
print(f"  Sharpe: {best_result['sharpe_ratio']:.2f}")
print(f"  Trades: {best_result['total_trades']}")
print(f"{'='*60}")

# 결과 저장
exp_config = {
    "strategy": "3-action-v2-tuning",
    "test_period": f"{cfg.TEST_START} ~ {cfg.DATA_END}",
    "best_config": {"lookahead": best_cfg[0], "dead_zone": best_cfg[1]},
    "grid_search": [
        {"la": r["lookahead"], "dz": r["dead_zone"], "return": r["model_return_pct"],
         "sharpe": r["sharpe_ratio"], "trades": r["total_trades"], "mdd": r["mdd_pct"]}
        for r in all_results
    ],
    "train_info": best_info,
}
exp_dir = create_experiment("lgbm_tuning_2024", exp_config)
save_metrics(exp_dir, best_result)
lgbm_model.save_model(best_model, os.path.join(exp_dir, "model.txt"))

# 그리드 서치 히트맵
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    fp = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="muted")

# 히트맵 데이터
res_df = pd.DataFrame(all_results)
for metric, title in [("model_return_pct", "Return (%)"), ("sharpe_ratio", "Sharpe"), ("total_trades", "Trades")]:
    pivot = res_df.pivot_table(index="dead_zone", columns="lookahead", values=metric)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = "RdYlGn" if metric != "total_trades" else "YlOrRd"
    sns.heatmap(pivot, annot=True, fmt=".1f" if metric != "total_trades" else ".0f",
                cmap=cmap, center=0 if metric == "model_return_pct" else None,
                ax=ax, linewidths=0.5)
    ax.set_title(f"Grid Search: {title}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Lookahead (hours)")
    ax.set_ylabel("Dead Zone")
    ax.set_yticklabels([f"{v*100:.1f}%" for v in pivot.index], rotation=0)
    plt.tight_layout()
    path = os.path.join(exp_dir, f"heatmap_{metric}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Chart] {path}")

# Best 모델 매매 타점
import matplotlib.dates as mdates
dates = best_test_df["date"].values
prices = best_test_df["close"].values
trade_log = best_result["trade_log"]

fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.plot(dates, prices, color="#333", linewidth=1, alpha=0.8, label="BTC/USDT")
bt = [dates[t[0]] for t in trade_log if t[1] == "BUY"]
bp = [t[2] for t in trade_log if t[1] == "BUY"]
st = [dates[t[0]] for t in trade_log if t[1] == "SELL"]
sp = [t[2] for t in trade_log if t[1] == "SELL"]
ax.scatter(bt, bp, marker="^", color="#2196F3", s=120, zorder=5,
           label=f"Buy ({len(bt)})", edgecolors="white", linewidths=0.5)
ax.scatter(st, sp, marker="v", color="#E53935", s=120, zorder=5,
           label=f"Sell ({len(st)})", edgecolors="white", linewidths=0.5)
for i in range(min(len(bt), len(st))):
    ax.axvspan(bt[i], st[i], alpha=0.06, color="#2196F3")
la, dz = best_cfg
ax.set_title(
    f"[Best] LA={la}h DZ={dz*100:.1f}%  |  "
    f"Return: {best_result['model_return_pct']:+.2f}%  vs  B&H: {best_result['buyhold_return_pct']:+.2f}%  |  "
    f"Sharpe: {best_result['sharpe_ratio']:.2f}  |  Trades: {best_result['total_trades']}",
    fontsize=13, fontweight="bold")
ax.set_ylabel("Price (USDT)")
ax.legend(fontsize=10, loc="upper left", frameon=True)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
sns.despine(ax=ax, left=True, bottom=True)

ax = axes[1]
pv = np.array(best_result["portfolio_values"])
pv_norm = pv / pv[0]
bh_norm = prices / prices[0]
pd2 = dates[:len(pv_norm)]
ax.plot(pd2, pv_norm, color="#2196F3", linewidth=2, label=f"Model ({best_result['model_return_pct']:+.2f}%)")
ax.plot(dates[:len(bh_norm)], bh_norm, color="#FF9800", linewidth=2, alpha=0.7, label=f"B&H ({best_result['buyhold_return_pct']:+.2f}%)")
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Portfolio")
ax.legend(fontsize=9, frameon=True)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
sns.despine(ax=ax, left=True, bottom=True)

plt.tight_layout()
path = os.path.join(exp_dir, "best_trade_points.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Chart] {path}")

# 매매 로그
print(f"\n[Trade Log - Best]")
for idx, (step, action, price) in enumerate(trade_log):
    date = best_test_df["date"].iloc[step]
    print(f"  {idx+1:3d}. {date.strftime('%Y-%m-%d %H:%M')} | {action:4s} | ${price:,.2f}")

print(f"\n[Saved] {exp_dir}")
