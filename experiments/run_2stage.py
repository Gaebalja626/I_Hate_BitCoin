"""
3-Class 분류 + 추가 피처 (Fear&Greed, Funding Rate) + TTA
LA sweep: 6h, 12h, 24h, 48h 비교
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
from fetch_extra_features import fetch_fear_greed, fetch_funding_rate, merge_extra_features
from experiment import create_experiment

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

# ── 설정 ──
DZ = 0.01
RETRAIN_EVERY = 720  # 1달마다 재학습
OOS_START = "2024-01-01"
OOS_END = "2025-01-01"

# ── 1) 데이터 + 추가 피처 ──
print("[STEP 1] Data + Extra Features")
raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, "2026-01-01", cfg.TECH_INDICATORS)
fg_df = fetch_fear_greed()
fr_df = fetch_funding_rate()
raw = merge_extra_features(raw, fg_df, fr_df)

# ── 2) LA sweep ──
la_configs = [6, 12, 24, 48]
all_sweep_results = []
colors = ["#2196F3", "#E53935", "#43A047", "#FF9800"]

exp_dir = create_experiment("3class_extra_tta", {
    "la_sweep": la_configs, "dz": DZ,
    "retrain_every": RETRAIN_EVERY,
    "extra_features": ["fear_greed", "funding_rate", "funding_rate_ma8",
                       "funding_rate_ma24", "funding_rate_cumsum_24h"],
})

fig_compare, ax_compare = plt.subplots(figsize=(18, 7))

for li, LA in enumerate(la_configs):
    print(f"\n{'='*60}")
    print(f"  LA={LA}h, DZ={DZ*100:.1f}%, Retrain/{RETRAIN_EVERY}h")
    print(f"{'='*60}")

    # 피처 생성
    df_all, fcols = create_features_v2(raw, cfg.TECH_INDICATORS, lookahead=LA, buy_threshold=DZ)
    df_all["date"] = pd.to_datetime(df_all["date"])

    extra_cols = ["fear_greed", "funding_rate", "funding_rate_ma8",
                  "funding_rate_ma24", "funding_rate_cumsum_24h"]
    for col in extra_cols:
        if col in df_all.columns:
            fcols.append(col)
    fcols = [c for c in fcols if c in df_all.columns]

    oos_start_dt = pd.Timestamp(OOS_START)
    oos_mask = (df_all["date"] >= oos_start_dt) & (df_all["date"] < pd.Timestamp(OOS_END))
    oos_indices = df_all[oos_mask].index.tolist()

    # TTA Loop
    all_actions = []
    retrain_count = 0
    retrain_points = []

    i = 0
    while i < len(oos_indices):
        chunk_end = min(i + RETRAIN_EVERY, len(oos_indices))
        chunk_indices = oos_indices[i:chunk_end]

        current_time = df_all.loc[chunk_indices[0], "date"]
        train_cutoff = current_time - pd.Timedelta(hours=LA)
        train_df = df_all[df_all["date"] <= train_cutoff].reset_index(drop=True)
        pred_df = df_all.loc[chunk_indices].reset_index(drop=True)

        retrain_count += 1
        retrain_points.append(current_time)

        model, feat_imp, info = lgbm_model.train_3class(train_df, fcols)
        actions, probs = lgbm_model.predict_3class(model, pred_df, fcols)
        all_actions.extend(actions.tolist())

        nb = int(sum(actions == 0))
        nh = int(sum(actions == 1))
        ns = int(sum(actions == 2))
        print(f"  [#{retrain_count}] {current_time.strftime('%Y-%m-%d')} | "
              f"iter={info['best_iteration']} | B={nb} H={nh} S={ns}")

        if retrain_count == 1:
            print(f"    [Top 10 Features]")
            for _, row in feat_imp.head(10).iterrows():
                print(f"      {row['feature']:28s} {row['importance']:.1f}")

        i = chunk_end

    # 백테스트
    oos_df = df_all.loc[oos_indices].reset_index(drop=True)
    actions_arr = np.array(all_actions)
    result = run_3action(actions_arr, oos_df)
    trade_log = result["trade_log"]

    nb_total = int(sum(actions_arr == 0))
    nh_total = int(sum(actions_arr == 1))
    ns_total = int(sum(actions_arr == 2))

    print(f"\n  [LA={LA}h] Return={result['model_return_pct']:+.2f}% | "
          f"B&H={result['buyhold_return_pct']:+.2f}% | "
          f"Sharpe={result['sharpe_ratio']:.2f} | MDD={result['mdd_pct']:.2f}% | "
          f"Trades={result['total_trades']} | B={nb_total} H={nh_total} S={ns_total}")

    all_sweep_results.append({
        "la": LA, "return": result["model_return_pct"],
        "bh": result["buyhold_return_pct"],
        "sharpe": result["sharpe_ratio"], "mdd": result["mdd_pct"],
        "trades": result["total_trades"],
        "buy": nb_total, "hold": nh_total, "sell": ns_total,
    })

    # 비교 차트에 추가
    pv = np.array(result["portfolio_values"])
    pv_norm = pv / pv[0]
    dates = oos_df["date"].values[:len(pv_norm)]
    label = f"LA={LA}h ({result['model_return_pct']:+.2f}%)"
    ax_compare.plot(dates, pv_norm, color=colors[li], linewidth=2, label=label)

    # 개별 매매 차트
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [3, 1]})
    prices = oos_df["close"].values
    all_dates = oos_df["date"].values

    ax = axes[0]
    ax.plot(all_dates, prices, color="#333", linewidth=1, alpha=0.8, label="BTC/USDT")
    bt = [all_dates[t[0]] for t in trade_log if t[1] == "BUY"]
    bp = [t[2] for t in trade_log if t[1] == "BUY"]
    st = [all_dates[t[0]] for t in trade_log if t[1] == "SELL"]
    sp = [t[2] for t in trade_log if t[1] == "SELL"]
    ax.scatter(bt, bp, marker="^", color="#2196F3", s=100, zorder=5,
               label=f"Buy ({len(bt)})", edgecolors="white", linewidths=0.5)
    ax.scatter(st, sp, marker="v", color="#E53935", s=100, zorder=5,
               label=f"Sell ({len(st)})", edgecolors="white", linewidths=0.5)
    for j in range(min(len(bt), len(st))):
        ax.axvspan(bt[j], st[j], alpha=0.06, color="#2196F3")
    for rt in retrain_points[1:]:
        ax.axvline(x=rt, color="#9C27B0", alpha=0.3, linestyle=":", linewidth=1)
    ax.set_title(
        f"[3-Class+Extra] LA={LA}h DZ={DZ*100:.1f}%  |  "
        f"Return: {result['model_return_pct']:+.2f}% vs B&H: {result['buyhold_return_pct']:+.2f}%  |  "
        f"Sharpe: {result['sharpe_ratio']:.2f} | Trades: {result['total_trades']}",
        fontsize=12, fontweight="bold")
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
    path = os.path.join(exp_dir, f"trades_LA{LA}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Chart] {path}")

    # 매매 로그
    for idx, (step, action, price) in enumerate(trade_log):
        date = oos_df["date"].iloc[step]
        print(f"    {idx+1:3d}. {date.strftime('%Y-%m-%d %H:%M')} | {action:4s} | ${price:,.2f}")

# B&H 기준선
bh_prices = oos_df["close"].values
bh_norm = bh_prices / bh_prices[0]
bh_ret = (bh_prices[-1] / bh_prices[0] - 1) * 100
ax_compare.plot(oos_df["date"].values[:len(bh_norm)], bh_norm, color="gray",
                linewidth=2.5, linestyle="--", alpha=0.8, label=f"B&H ({bh_ret:+.2f}%)")
ax_compare.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
ax_compare.set_title("3-Class + Extra Features + TTA: LA Sweep (2025~2026)",
                     fontsize=14, fontweight="bold")
ax_compare.set_ylabel("Portfolio (normalized)")
ax_compare.legend(fontsize=11, loc="upper left", frameon=True, fancybox=True, shadow=True)
ax_compare.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_compare.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax_compare.xaxis.get_majorticklabels(), rotation=45)
sns.despine(ax=ax_compare, left=True, bottom=True)
fig_compare.tight_layout()
path = os.path.join(exp_dir, "la_sweep_comparison.png")
fig_compare.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig_compare)
print(f"\n[Chart] {path}")

# 최종 요약
print(f"\n{'='*80}")
print(f"{'LA':>6} | {'Return':>10} | {'B&H':>8} | {'MDD':>8} | {'Sharpe':>7} | {'Trades':>6} | {'Buy':>5} {'Hold':>5} {'Sell':>5}")
print("-" * 80)
for r in all_sweep_results:
    print(f"  {r['la']:>3}h | {r['return']:>+9.2f}% | {r['bh']:>+7.2f}% | "
          f"{r['mdd']:>7.2f}% | {r['sharpe']:>6.2f} | {r['trades']:>6} | "
          f"{r['buy']:>5} {r['hold']:>5} {r['sell']:>5}")
print(f"{'='*80}")

import json
with open(os.path.join(exp_dir, "summary.json"), "w") as f:
    json.dump({"sweep": all_sweep_results}, f, indent=2)
print(f"\n[Saved] {exp_dir}")
