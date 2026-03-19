"""
Ablation Study: TTA vs No-TTA
- LA=24h, DZ=1%
- WITH TTA: 월간 재학습 (720h 간격)
- WITHOUT TTA: 단일 정적 모델 (OOS 시작 전까지의 데이터로 1회 학습)
- Fear&Greed 1일 shift 적용
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
LA = 24
DZ = 0.01
RETRAIN_EVERY = 720

# ── 1) 데이터 ──
print("[STEP 1] Data + Extra Features")
raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, "2026-01-01", cfg.TECH_INDICATORS)
fg_df = fetch_fear_greed()
fr_df = fetch_funding_rate()
raw = merge_extra_features(raw, fg_df, fr_df)

# ── 2) 피처 ──
print(f"[STEP 2] Features v2 (LA={LA}h, DZ={DZ*100:.0f}%)")
df_all, fcols = create_features_v2(raw, cfg.TECH_INDICATORS, lookahead=LA, buy_threshold=DZ)
df_all["date"] = pd.to_datetime(df_all["date"])

extra_cols = ["fear_greed", "funding_rate", "funding_rate_ma8",
              "funding_rate_ma24", "funding_rate_cumsum_24h"]
for col in extra_cols:
    if col in df_all.columns:
        fcols.append(col)
fcols = [c for c in fcols if c in df_all.columns]

# ── 3) 두 기간 × 두 모드 실험 ──
periods = [
    ("2024-01-01", "2025-01-01", "2024~2025"),
    ("2025-01-01", "2026-01-01", "2025~2026"),
]

exp_dir = create_experiment("ablation_tta", {
    "la": LA, "dz": DZ, "retrain_every": RETRAIN_EVERY,
    "description": "TTA vs No-TTA ablation study",
})

all_results = []

for oos_start, oos_end, label in periods:
    print(f"\n{'='*70}")
    print(f"  OOS: {label}")
    print(f"{'='*70}")

    oos_mask = (df_all["date"] >= pd.Timestamp(oos_start)) & (df_all["date"] < pd.Timestamp(oos_end))
    oos_indices = df_all[oos_mask].index.tolist()
    oos_df = df_all.loc[oos_indices].reset_index(drop=True)

    # ────────────────────────────────
    # (A) WITH TTA (월간 재학습)
    # ────────────────────────────────
    print(f"\n  --- [A] WITH TTA (retrain every {RETRAIN_EVERY}h) ---")
    all_actions_tta = []
    retrain_count = 0

    i = 0
    while i < len(oos_indices):
        chunk_end = min(i + RETRAIN_EVERY, len(oos_indices))
        chunk_indices = oos_indices[i:chunk_end]

        current_time = df_all.loc[chunk_indices[0], "date"]
        train_cutoff = current_time - pd.Timedelta(hours=LA)
        train_df = df_all[df_all["date"] <= train_cutoff].reset_index(drop=True)
        pred_df = df_all.loc[chunk_indices].reset_index(drop=True)

        retrain_count += 1
        model, feat_imp, info = lgbm_model.train_3class(train_df, fcols)
        actions, probs = lgbm_model.predict_3class(model, pred_df, fcols)
        all_actions_tta.extend(actions.tolist())

        nb = int(sum(actions == 0))
        nh = int(sum(actions == 1))
        ns = int(sum(actions == 2))
        print(f"    [#{retrain_count}] {current_time.strftime('%Y-%m-%d')} | "
              f"iter={info['best_iteration']} | B={nb} H={nh} S={ns}")

        i = chunk_end

    actions_tta = np.array(all_actions_tta)
    result_tta = run_3action(actions_tta, oos_df)

    print(f"\n  [TTA] Return={result_tta['model_return_pct']:+.2f}% | "
          f"B&H={result_tta['buyhold_return_pct']:+.2f}% | "
          f"Sharpe={result_tta['sharpe_ratio']:.2f} | MDD={result_tta['mdd_pct']:.2f}% | "
          f"Trades={result_tta['total_trades']}")

    # ────────────────────────────────
    # (B) WITHOUT TTA (단일 정적 모델)
    # ────────────────────────────────
    print(f"\n  --- [B] WITHOUT TTA (single static model) ---")

    # OOS 시작 전까지의 데이터로 1회 학습
    train_cutoff_static = pd.Timestamp(oos_start) - pd.Timedelta(hours=LA)
    train_df_static = df_all[df_all["date"] <= train_cutoff_static].reset_index(drop=True)

    print(f"    Train data: {len(train_df_static)} samples (up to {train_cutoff_static})")
    model_static, feat_imp_static, info_static = lgbm_model.train_3class(train_df_static, fcols)
    actions_static, probs_static = lgbm_model.predict_3class(model_static, oos_df, fcols)

    result_no_tta = run_3action(actions_static, oos_df)

    nb = int(sum(actions_static == 0))
    nh = int(sum(actions_static == 1))
    ns = int(sum(actions_static == 2))
    print(f"\n  [No TTA] Return={result_no_tta['model_return_pct']:+.2f}% | "
          f"B&H={result_no_tta['buyhold_return_pct']:+.2f}% | "
          f"Sharpe={result_no_tta['sharpe_ratio']:.2f} | MDD={result_no_tta['mdd_pct']:.2f}% | "
          f"Trades={result_no_tta['total_trades']} | B={nb} H={nh} S={ns}")

    all_results.append({
        "period": label,
        "tta_return": result_tta["model_return_pct"],
        "no_tta_return": result_no_tta["model_return_pct"],
        "bh": result_tta["buyhold_return_pct"],
        "tta_sharpe": result_tta["sharpe_ratio"],
        "no_tta_sharpe": result_no_tta["sharpe_ratio"],
        "tta_mdd": result_tta["mdd_pct"],
        "no_tta_mdd": result_no_tta["mdd_pct"],
        "tta_trades": result_tta["total_trades"],
        "no_tta_trades": result_no_tta["total_trades"],
    })

    # ── 비교 차트 ──
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [3, 1]})

    # 상단: 가격 + 매매 포인트
    ax = axes[0]
    prices = oos_df["close"].values
    dates = oos_df["date"].values
    ax.plot(dates, prices, color="#333", linewidth=1, alpha=0.8, label="BTC/USDT")
    ax.set_title(f"[{label}] TTA vs No-TTA | LA={LA}h DZ={DZ*100:.0f}%",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Price (USDT)")
    ax.legend(fontsize=10, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    sns.despine(ax=ax, left=True, bottom=True)

    # 하단: 포트폴리오 비교
    ax = axes[1]
    pv_tta = np.array(result_tta["portfolio_values"])
    pv_no = np.array(result_no_tta["portfolio_values"])
    pv_tta_norm = pv_tta / pv_tta[0]
    pv_no_norm = pv_no / pv_no[0]
    bh_norm = prices / prices[0]

    plot_dates_tta = dates[:len(pv_tta_norm)]
    plot_dates_no = dates[:len(pv_no_norm)]

    ax.plot(plot_dates_tta, pv_tta_norm, color="#2196F3", linewidth=2,
            label=f"TTA ({result_tta['model_return_pct']:+.2f}%)")
    ax.plot(plot_dates_no, pv_no_norm, color="#E53935", linewidth=2,
            label=f"No TTA ({result_no_tta['model_return_pct']:+.2f}%)")
    ax.plot(dates[:len(bh_norm)], bh_norm, color="#FF9800", linewidth=1.5, alpha=0.7,
            label=f"B&H ({result_tta['buyhold_return_pct']:+.2f}%)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Portfolio (normalized)")
    ax.legend(fontsize=9, frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    path = os.path.join(exp_dir, f"ablation_tta_{label.replace('~','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Chart] {path}")

# ── 최종 요약 ──
print(f"\n{'='*90}")
print(f"{'Period':>12} | {'TTA':>10} | {'No TTA':>10} | {'B&H':>8} | "
      f"{'TTA Sharpe':>10} | {'No TTA Sharpe':>13} | {'TTA MDD':>8} | {'No TTA MDD':>10}")
print("-" * 90)
for r in all_results:
    print(f"  {r['period']:>10} | {r['tta_return']:>+9.2f}% | {r['no_tta_return']:>+9.2f}% | "
          f"{r['bh']:>+7.2f}% | {r['tta_sharpe']:>9.2f} | {r['no_tta_sharpe']:>12.2f} | "
          f"{r['tta_mdd']:>7.2f}% | {r['no_tta_mdd']:>9.2f}%")
print(f"{'='*90}")

import json
with open(os.path.join(exp_dir, "summary.json"), "w") as f:
    json.dump({"results": all_results}, f, indent=2)
print(f"\n[Saved] {exp_dir}")
