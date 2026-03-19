"""
Test-Time Adaptation: 주기적 재학습으로 시장 변화에 적응
- 매 RETRAIN_EVERY 시간마다 모델 재학습 (새로 쌓인 GT 포함)
- LA=48h이므로 48시간 전까지의 데이터는 라벨 확정
"""
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
LA = 48
DZ = 0.01
RETRAIN_EVERY = 720  # 1달마다 재학습
OOS_START = "2025-01-01"
OOS_END = "2026-01-01"

print(f"[Config] LA={LA}h, DZ={DZ*100:.1f}%, Retrain every {RETRAIN_EVERY}h")

# ── 데이터 ──
print("[STEP 1] Data")
train_raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, cfg.DATA_END, cfg.TECH_INDICATORS)
test_raw = fetch_btc_data(cfg.PAIR, cfg.TIMEFRAME, OOS_START, OOS_END)
test_raw = add_technical_indicators(test_raw, cfg.TECH_INDICATORS)
full_raw = pd.concat([train_raw, test_raw]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# 전체 피처 생성 (라벨은 미래 데이터 필요하므로 전체로 만든 뒤 시간 기준 split)
print("[STEP 2] Features")
df_all, fcols = create_features_v2(full_raw, cfg.TECH_INDICATORS, lookahead=LA, buy_threshold=DZ)
df_all["date"] = pd.to_datetime(df_all["date"])

oos_start_dt = pd.Timestamp(OOS_START)
oos_end_dt = pd.Timestamp(OOS_END)

# OOS 구간 인덱스
oos_mask = (df_all["date"] >= oos_start_dt) & (df_all["date"] < oos_end_dt)
oos_indices = df_all[oos_mask].index.tolist()
print(f"  OOS bars: {len(oos_indices)}")

# ── TTA Loop ──
print(f"\n[STEP 3] Test-Time Adaptation (retrain every {RETRAIN_EVERY}h)")
all_actions = []
all_oos_dates = []
retrain_count = 0
retrain_points = []

i = 0
while i < len(oos_indices):
    # 이번 chunk: i ~ i+RETRAIN_EVERY
    chunk_end = min(i + RETRAIN_EVERY, len(oos_indices))
    chunk_indices = oos_indices[i:chunk_end]

    # 학습 데이터: OOS 시작 전 데이터 + 이미 지나간 OOS 데이터 (GT 확정분)
    # 현재 시점에서 LA시간 전까지는 라벨이 확정됨
    current_time = df_all.loc[chunk_indices[0], "date"]
    train_cutoff = current_time - pd.Timedelta(hours=LA)
    train_df = df_all[df_all["date"] <= train_cutoff].reset_index(drop=True)

    # 예측 대상
    pred_df = df_all.loc[chunk_indices].reset_index(drop=True)

    retrain_count += 1
    retrain_points.append(current_time)
    print(f"\n  [Retrain #{retrain_count}] {current_time.strftime('%Y-%m-%d %H:%M')}"
          f" | Train: {len(train_df)} rows | Predict: {len(pred_df)} bars")

    # 학습
    model, _, info = lgbm_model.train_3class(train_df, fcols)
    print(f"    best_iter={info['best_iteration']} | val_loss={info['best_score']:.4f}")

    # 예측
    actions, probs = lgbm_model.predict_3class(model, pred_df, fcols)
    all_actions.extend(actions.tolist())
    all_oos_dates.extend(pred_df["date"].tolist())

    nb = int(sum(actions == 0))
    nh = int(sum(actions == 1))
    ns = int(sum(actions == 2))
    print(f"    Actions: Buy={nb}, Hold={nh}, Sell={ns}")

    i = chunk_end

print(f"\n  Total retrains: {retrain_count}")

# ── 백테스트 ──
print("\n[STEP 4] Backtest")
oos_df = df_all.loc[oos_indices].reset_index(drop=True)
actions_arr = np.array(all_actions)
result = run_3action(actions_arr, oos_df)
trade_log = result["trade_log"]

nb_total = int(sum(actions_arr == 0))
nh_total = int(sum(actions_arr == 1))
ns_total = int(sum(actions_arr == 2))

print(f"""
{'='*60}
[TTA] LA=48h DZ=0.5% | Retrain every {RETRAIN_EVERY}h
{'='*60}
  Return:  {result['model_return_pct']:+.2f}%
  B&H:    {result['buyhold_return_pct']:+.2f}%
  MDD:    {result['mdd_pct']:.2f}%
  Sharpe: {result['sharpe_ratio']:.2f}
  Trades: {result['total_trades']}
  Actions: Buy={nb_total}, Hold={nh_total}, Sell={ns_total}
  Retrains: {retrain_count}
{'='*60}
""")

# ── 비교용: 재학습 없는 baseline ──
print("[STEP 5] Baseline (no retrain)")
base_train = df_all[df_all["date"] < oos_start_dt].reset_index(drop=True)
base_model, _, base_info = lgbm_model.train_3class(base_train, fcols)
base_actions, _ = lgbm_model.predict_3class(base_model, oos_df, fcols)
base_result = run_3action(base_actions, oos_df)
print(f"  Baseline Return: {base_result['model_return_pct']:+.2f}% | "
      f"Sharpe: {base_result['sharpe_ratio']:.2f} | Trades: {base_result['total_trades']}")

# ── 실험 저장 ──
exp_dir = create_experiment("tta_weekly", {
    "config": {"lookahead": LA, "dead_zone": DZ, "retrain_every": RETRAIN_EVERY},
    "period": f"{OOS_START} ~ {OOS_END}",
    "retrains": retrain_count,
})

# ── 차트 1: 매매 타점 ──
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
# 재학습 시점 표시
for rt in retrain_points[1:]:  # 첫 번째는 시작점이라 제외
    ax.axvline(x=rt, color="#9C27B0", alpha=0.3, linestyle=":", linewidth=1)
ax.set_title(
    f"[TTA] LA=48h DZ=0.5% Retrain/{RETRAIN_EVERY}h  |  "
    f"Return: {result['model_return_pct']:+.2f}%  vs  B&H: {result['buyhold_return_pct']:+.2f}%  |  "
    f"Sharpe: {result['sharpe_ratio']:.2f}  |  Trades: {result['total_trades']}",
    fontsize=12, fontweight="bold")
ax.set_ylabel("Price (USDT)")
ax.legend(fontsize=10, loc="upper left", frameon=True)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
sns.despine(ax=ax, left=True, bottom=True)

ax = axes[1]
pv = np.array(result["portfolio_values"])
pv_norm = pv / pv[0]
bh_norm = prices / prices[0]
# baseline
bpv = np.array(base_result["portfolio_values"])
bpv_norm = bpv / bpv[0]

plot_dates = all_dates[:len(pv_norm)]
ax.plot(plot_dates, pv_norm, color="#2196F3", linewidth=2,
        label=f"TTA ({result['model_return_pct']:+.2f}%)")
ax.plot(all_dates[:len(bpv_norm)], bpv_norm, color="#9C27B0", linewidth=1.5, alpha=0.7,
        label=f"No-Retrain ({base_result['model_return_pct']:+.2f}%)")
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
path = os.path.join(exp_dir, "tta_trades.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Chart] {path}")

# ── 차트 2: TTA vs Baseline vs B&H 비교 ──
fig, ax = plt.subplots(figsize=(18, 7))
ax.plot(plot_dates, pv_norm, color="#2196F3", linewidth=2.5,
        label=f"TTA Retrain/{RETRAIN_EVERY}h ({result['model_return_pct']:+.2f}%)")
ax.plot(all_dates[:len(bpv_norm)], bpv_norm, color="#9C27B0", linewidth=2, alpha=0.8,
        label=f"Static Model ({base_result['model_return_pct']:+.2f}%)")
ax.plot(all_dates[:len(bh_norm)], bh_norm, color="#FF9800", linewidth=2.5, linestyle="--",
        alpha=0.8, label=f"Buy & Hold ({result['buyhold_return_pct']:+.2f}%)")
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
for rt in retrain_points[1:]:
    ax.axvline(x=rt, color="#9C27B0", alpha=0.15, linestyle=":", linewidth=1)
ax.set_title(
    f"Test-Time Adaptation vs Static Model vs B&H  |  2025~2026",
    fontsize=14, fontweight="bold")
ax.set_ylabel("Portfolio (normalized)")
ax.legend(fontsize=11, loc="upper left", frameon=True, fancybox=True, shadow=True)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
sns.despine(ax=ax, left=True, bottom=True)
plt.tight_layout()
path = os.path.join(exp_dir, "tta_vs_baseline.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Chart] {path}")

# ── 매매 로그 ──
print(f"\n[Trade Log - TTA]")
for idx, (step, action, price) in enumerate(trade_log):
    date = oos_df["date"].iloc[step]
    print(f"  {idx+1:3d}. {date.strftime('%Y-%m-%d %H:%M')} | {action:4s} | ${price:,.2f}")

# ── 요약 저장 ──
import json
summary = {
    "tta": {
        "return": result["model_return_pct"],
        "bh": result["buyhold_return_pct"],
        "mdd": result["mdd_pct"],
        "sharpe": result["sharpe_ratio"],
        "trades": result["total_trades"],
        "retrains": retrain_count,
    },
    "baseline": {
        "return": base_result["model_return_pct"],
        "sharpe": base_result["sharpe_ratio"],
        "trades": base_result["total_trades"],
    },
}
with open(os.path.join(exp_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[Saved] {exp_dir}")
