"""
Walk-forward 검증: 2023~2024 튜닝 → 2024~2025 OOS 평가
"""
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

# 데이터 (2020~2025)
print("[STEP 1] Data")
raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, cfg.DATA_END, cfg.TECH_INDICATORS)

# ============================================================
# Phase 1: 2023~2024 튜닝 (train: 2020~2023, test: 2023~2024)
# ============================================================
print("\n" + "=" * 60)
print("  Phase 1: Grid Search on 2023~2024")
print("=" * 60)

configs = []
for la in [6, 12, 24, 48]:
    for dz in [0.003, 0.005, 0.007, 0.01, 0.015, 0.02]:
        configs.append((la, dz))

header = f"{'LA':>4} | {'DZ':>6} | {'Iter':>5} | {'Return':>8} | {'MDD':>8} | {'Sharpe':>7} | {'Trades':>6}"
print(header)
print("-" * len(header))

best_ret = -999
best_sharpe = -999
all_tuning = []

for la, dz in configs:
    df, fcols = create_features_v2(raw, cfg.TECH_INDICATORS, lookahead=la, buy_threshold=dz)
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[df["date"] < "2023-06-01"].reset_index(drop=True)
    test_df = df[(df["date"] >= "2023-06-01") & (df["date"] < "2024-06-01")].reset_index(drop=True)

    if len(test_df) < 100 or len(train_df) < 1000:
        continue

    model, _, info = lgbm_model.train_3class(train_df, fcols)
    actions, _ = lgbm_model.predict_3class(model, test_df, fcols)
    result = run_3action(actions, test_df)

    ret = result["model_return_pct"]
    row = {"la": la, "dz": dz, "return": ret, "sharpe": result["sharpe_ratio"],
           "mdd": result["mdd_pct"], "trades": result["total_trades"],
           "bh": result["buyhold_return_pct"], "iter": info["best_iteration"]}
    all_tuning.append(row)

    if ret > best_ret and result["total_trades"] >= 5:
        best_ret = ret
        best_cfg = (la, dz)

    print(f"{la:>3}h | {dz*100:>5.1f}% | {info['best_iteration']:>5} | "
          f"{ret:>+7.2f}% | {result['mdd_pct']:>7.2f}% | "
          f"{result['sharpe_ratio']:>6.2f} | {result['total_trades']:>6}")

# Top 5 (거래 5회 이상)
tuning_active = [r for r in all_tuning if r["trades"] >= 5]
tuning_active.sort(key=lambda x: x["return"], reverse=True)
top5 = tuning_active[:5]

print(f"\n[Top 5 on 2023~2024 (trades >= 5)]")
for i, r in enumerate(top5):
    print(f"  {i+1}. LA={r['la']}h DZ={r['dz']*100:.1f}%: {r['return']:+.2f}% | Sharpe={r['sharpe']:.2f} | Trades={r['trades']}")

# ============================================================
# Phase 2: Top 5를 2024~2025 OOS에서 평가
# ============================================================
print("\n" + "=" * 60)
print("  Phase 2: OOS Validation on 2024~2025")
print("=" * 60)

colors = ["#2196F3", "#E53935", "#43A047", "#FF9800", "#9C27B0"]
exp_dir = create_experiment("walkforward_2023_2024", {
    "tuning_period": "2023-06 ~ 2024-06",
    "oos_period": "2024-06 ~ 2025-01",
    "top5": [{"la": r["la"], "dz": r["dz"], "tuning_return": r["return"]} for r in top5],
})

fig_compare, ax_compare = plt.subplots(figsize=(18, 7))
oos_results = []

for i, cfg_row in enumerate(top5):
    la, dz = cfg_row["la"], cfg_row["dz"]
    print(f"\n[Config {i+1}] LA={la}h DZ={dz*100:.1f}% (tuning: {cfg_row['return']:+.2f}%)")

    df, fcols = create_features_v2(raw, cfg.TECH_INDICATORS, lookahead=la, buy_threshold=dz)
    df["date"] = pd.to_datetime(df["date"])
    # 학습: 2020~2024-06, 테스트: 2024-06~2025-01
    train_df = df[df["date"] < "2024-06-01"].reset_index(drop=True)
    test_df = df[(df["date"] >= "2024-06-01") & (df["date"] < "2025-01-01")].reset_index(drop=True)

    model, feat_imp, info = lgbm_model.train_3class(train_df, fcols)
    actions, probs = lgbm_model.predict_3class(model, test_df, fcols)
    result = run_3action(actions, test_df)
    trade_log = result["trade_log"]

    nb = int(sum(actions == 0))
    nh = int(sum(actions == 1))
    ns = int(sum(actions == 2))

    print(f"  OOS Return: {result['model_return_pct']:+.2f}% | B&H: {result['buyhold_return_pct']:+.2f}%")
    print(f"  MDD: {result['mdd_pct']:.2f}% | Sharpe: {result['sharpe_ratio']:.2f} | Trades: {result['total_trades']}")
    print(f"  Actions: Buy={nb}, Hold={nh}, Sell={ns}")

    oos_results.append({
        "la": la, "dz": dz,
        "tuning_ret": cfg_row["return"],
        "oos_ret": result["model_return_pct"],
        "bh": result["buyhold_return_pct"],
        "mdd": result["mdd_pct"],
        "sharpe": result["sharpe_ratio"],
        "trades": result["total_trades"],
    })

    # 비교 차트에 추가
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
        f"[WF OOS #{i+1}] LA={la}h DZ={dz*100:.1f}%  |  "
        f"Tuning: {cfg_row['return']:+.2f}%  ->  OOS: {result['model_return_pct']:+.2f}%  |  "
        f"B&H: {result['buyhold_return_pct']:+.2f}%  |  Trades: {result['total_trades']}",
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
    path = os.path.join(exp_dir, f"wf_config{i+1}_LA{la}_DZ{int(dz*1000)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Chart] {path}")

    # 매매 로그
    for idx, (step, action, price) in enumerate(trade_log):
        date = test_df["date"].iloc[step]
        print(f"    {idx+1:3d}. {date.strftime('%Y-%m-%d %H:%M')} | {action:4s} | ${price:,.2f}")

# B&H 기준선
bh_df = df[(df["date"] >= "2024-06-01") & (df["date"] < "2025-01-01")]
bh_prices = bh_df["close"].values
bh_dates = bh_df["date"].values
bh_norm = bh_prices / bh_prices[0]
bh_ret = (bh_prices[-1] / bh_prices[0] - 1) * 100
ax_compare.plot(bh_dates[:len(bh_norm)], bh_norm, color="#FF9800", linewidth=2.5,
                linestyle="--", alpha=0.8, label=f"Buy&Hold ({bh_ret:+.2f}%)")
ax_compare.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
ax_compare.set_title("Walk-Forward: Tuned on 2023~2024 -> OOS 2024~2025", fontsize=14, fontweight="bold")
ax_compare.set_ylabel("Portfolio (normalized)")
ax_compare.legend(fontsize=9, loc="upper left", frameon=True, fancybox=True, shadow=True)
ax_compare.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_compare.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax_compare.xaxis.get_majorticklabels(), rotation=45)
sns.despine(ax=ax_compare, left=True, bottom=True)
fig_compare.tight_layout()
path = os.path.join(exp_dir, "wf_top5_comparison.png")
fig_compare.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig_compare)
print(f"\n[Chart] {path}")

# 최종 요약
print(f"\n{'='*90}")
print(f"{'Config':>20} | {'Tuning':>10} | {'OOS Return':>10} | {'B&H':>8} | {'MDD':>8} | {'Sharpe':>7} | {'Trades':>6}")
print("-" * 90)
for r in oos_results:
    c = f"LA={r['la']}h DZ={r['dz']*100:.1f}%"
    print(f"{c:>20} | {r['tuning_ret']:>+9.2f}% | {r['oos_ret']:>+9.2f}% | "
          f"{r['bh']:>+7.2f}% | {r['mdd']:>7.2f}% | {r['sharpe']:>6.2f} | {r['trades']:>6}")
print(f"{'='*90}")

# 메트릭 저장
import json
with open(os.path.join(exp_dir, "walkforward_summary.json"), "w") as f:
    json.dump({"tuning": all_tuning, "oos": oos_results}, f, indent=2)

print(f"\n[Saved] {exp_dir}")
