"""2024~2025 튜닝 Top 5 설정을 OOS(2025~2026)에서 검증 + 차트"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

os.environ["SSL_CERT_FILE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"
os.environ["REQUESTS_CA_BUNDLE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"

import config as cfg
from data_collector import load_or_fetch, fetch_btc_data, add_technical_indicators
from models import lgbm_model
from backtester import run_3action
from feature_engineer import create_features_v2
from experiment import create_experiment, save_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import seaborn as sns

font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    fp = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="muted")

# 데이터
print("[STEP 1] Data")
train_raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, cfg.DATA_END, cfg.TECH_INDICATORS)
test_raw = fetch_btc_data(cfg.PAIR, cfg.TIMEFRAME, "2025-01-01", "2026-01-01")
test_raw = add_technical_indicators(test_raw, cfg.TECH_INDICATORS)
full_raw = pd.concat([train_raw, test_raw]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# Top 5 (거래 5회 이상만, 2024~2025 기준)
top5 = [
    (6,  0.01,  "+45.10%", 10),   # 2024: +45.10%, Sharpe 2.48
    (48, 0.01,  "+32.53%", 108),  # 2024: +32.53%, Sharpe 2.14
    (48, 0.015, "+31.25%", 42),   # 2024: +31.25%, Sharpe 1.98
    (48, 0.005, "+31.05%", 30),   # 2024: +31.05%, Sharpe 2.25
    (48, 0.007, "+20.25%", 100),  # 2024: +20.25%, Sharpe 1.50
]

colors = ["#2196F3", "#E53935", "#43A047", "#FF9800", "#9C27B0"]
exp_dir = create_experiment("lgbm_top5_oos", {"configs": [{"la": la, "dz": dz} for la, dz, _, _ in top5]})

# 큰 비교 차트
fig_compare, ax_compare = plt.subplots(figsize=(18, 7))
all_oos_results = []

for i, (la, dz, in_sample_ret, in_sample_trades) in enumerate(top5):
    print(f"\n[Config {i+1}] LA={la}h, DZ={dz*100:.1f}%  (in-sample: {in_sample_ret}, {in_sample_trades} trades)")

    df, fcols = create_features_v2(full_raw, cfg.TECH_INDICATORS, lookahead=la, buy_threshold=dz)
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[df["date"] < "2025-01-01"].reset_index(drop=True)
    test_df = df[df["date"] >= "2025-01-01"].reset_index(drop=True)

    model, feat_imp, info = lgbm_model.train_3class(train_df, fcols)
    actions, probs = lgbm_model.predict_3class(model, test_df, fcols)
    result = run_3action(actions, test_df)
    trade_log = result["trade_log"]

    result["config"] = {"lookahead": la, "dead_zone": dz}
    result["in_sample_return"] = in_sample_ret
    all_oos_results.append(result)

    nb = int(sum(actions == 0))
    nh = int(sum(actions == 1))
    ns = int(sum(actions == 2))

    print(f"  OOS Return: {result['model_return_pct']:+.2f}%  |  B&H: {result['buyhold_return_pct']:+.2f}%")
    print(f"  MDD: {result['mdd_pct']:.2f}%  |  Sharpe: {result['sharpe_ratio']:.2f}  |  Trades: {result['total_trades']}")
    print(f"  Actions: Buy={nb}, Hold={nh}, Sell={ns}")

    # 포트폴리오 비교 차트에 추가
    pv = np.array(result["portfolio_values"])
    pv_norm = pv / pv[0]
    dates = test_df["date"].values[:len(pv_norm)]
    label = f"LA={la}h DZ={dz*100:.1f}% ({result['model_return_pct']:+.2f}%)"
    ax_compare.plot(dates, pv_norm, color=colors[i], linewidth=2, label=label)

    # 개별 매매 타점 차트
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [3, 1]})
    prices = test_df["close"].values
    all_dates = test_df["date"].values

    ax = axes[0]
    ax.plot(all_dates, prices, color="#333", linewidth=1, alpha=0.8, label="BTC/USDT")
    bt = [all_dates[t[0]] for t in trade_log if t[1] == "BUY"]
    bp = [t[2] for t in trade_log if t[1] == "BUY"]
    st = [all_dates[t[0]] for t in trade_log if t[1] == "SELL"]
    sp = [t[2] for t in trade_log if t[1] == "SELL"]
    ax.scatter(bt, bp, marker="^", color="#2196F3", s=120, zorder=5,
               label=f"Buy ({len(bt)})", edgecolors="white", linewidths=0.5)
    ax.scatter(st, sp, marker="v", color="#E53935", s=120, zorder=5,
               label=f"Sell ({len(st)})", edgecolors="white", linewidths=0.5)
    for j in range(min(len(bt), len(st))):
        ax.axvspan(bt[j], st[j], alpha=0.06, color="#2196F3")
    ax.set_title(
        f"[OOS #{i+1}] LA={la}h DZ={dz*100:.1f}%  |  "
        f"Return: {result['model_return_pct']:+.2f}%  vs  B&H: {result['buyhold_return_pct']:+.2f}%  |  "
        f"Sharpe: {result['sharpe_ratio']:.2f}  |  Trades: {result['total_trades']}",
        fontsize=13, fontweight="bold")
    ax.set_ylabel("Price (USDT)")
    ax.legend(fontsize=10, loc="upper left", frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    sns.despine(ax=ax, left=True, bottom=True)

    ax = axes[1]
    bh_norm = prices / prices[0]
    plot_dates = all_dates[:len(pv_norm)]
    ax.plot(plot_dates, pv_norm, color="#2196F3", linewidth=2,
            label=f"Model ({result['model_return_pct']:+.2f}%)")
    ax.plot(all_dates[:len(bh_norm)], bh_norm, color="#FF9800", linewidth=2, alpha=0.7,
            label=f"B&H ({result['buyhold_return_pct']:+.2f}%)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Portfolio")
    ax.legend(fontsize=9, frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    path = os.path.join(exp_dir, f"oos_config{i+1}_LA{la}_DZ{int(dz*1000)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Chart] {path}")

    # 매매 로그
    for idx, (step, action, price) in enumerate(trade_log):
        date = test_df["date"].iloc[step]
        print(f"    {idx+1:3d}. {date.strftime('%Y-%m-%d %H:%M')} | {action:4s} | ${price:,.2f}")

# B&H 기준선
prices0 = full_raw[full_raw["date"] >= "2025-01-01"]["close"].values
bh_dates = full_raw[full_raw["date"] >= "2025-01-01"]["date"].values[:len(prices0)]
bh_norm = prices0 / prices0[0]
bh_ret = (prices0[-1] / prices0[0] - 1) * 100
ax_compare.plot(bh_dates[:len(bh_norm)], bh_norm, color="#FF9800", linewidth=2.5,
                linestyle="--", alpha=0.8, label=f"Buy&Hold ({bh_ret:+.2f}%)")
ax_compare.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
ax_compare.set_title("[OOS 2025~2026] Top 5 Configs Comparison", fontsize=14, fontweight="bold")
ax_compare.set_ylabel("Portfolio (normalized)")
ax_compare.legend(fontsize=9, loc="upper left", frameon=True, fancybox=True, shadow=True)
ax_compare.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_compare.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax_compare.xaxis.get_majorticklabels(), rotation=45)
sns.despine(ax=ax_compare, left=True, bottom=True)
fig_compare.tight_layout()
path = os.path.join(exp_dir, "oos_top5_comparison.png")
fig_compare.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig_compare)
print(f"\n[Chart] {path}")

# 최종 요약
print(f"\n{'='*80}")
print(f"{'Config':>20} | {'In-Sample':>10} | {'OOS Return':>10} | {'B&H':>8} | {'MDD':>8} | {'Sharpe':>7} | {'Trades':>6}")
print("-" * 80)
for i, (la, dz, is_ret, is_tr) in enumerate(top5):
    r = all_oos_results[i]
    cfg_str = f"LA={la}h DZ={dz*100:.1f}%"
    print(f"{cfg_str:>20} | {is_ret:>10} | {r['model_return_pct']:>+9.2f}% | {r['buyhold_return_pct']:>+7.2f}% | {r['mdd_pct']:>7.2f}% | {r['sharpe_ratio']:>6.2f} | {r['total_trades']:>6}")
print(f"{'='*80}")

# 메트릭 저장
summary = {
    "experiment": "top5_oos_validation",
    "configs": [],
}
for i, (la, dz, is_ret, _) in enumerate(top5):
    r = all_oos_results[i]
    summary["configs"].append({
        "rank": i + 1,
        "lookahead": la,
        "dead_zone": dz,
        "in_sample_return": is_ret,
        "oos_return": r["model_return_pct"],
        "oos_bh": r["buyhold_return_pct"],
        "oos_mdd": r["mdd_pct"],
        "oos_sharpe": r["sharpe_ratio"],
        "oos_trades": r["total_trades"],
    })

import json
with open(os.path.join(exp_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[Saved] {exp_dir}")
