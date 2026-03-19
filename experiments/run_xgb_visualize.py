"""
XGBoost base 모델 매매 타점 + 포트폴리오 시각화
- 2024~2025 / 2025~2026 각각
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

os.environ["SSL_CERT_FILE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"
os.environ["REQUESTS_CA_BUNDLE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"

import xgboost as xgb

import config as cfg
from data_collector import load_or_fetch
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

LA = 24
DZ = 0.01
RETRAIN_EVERY = 720
VAL_RATIO = 0.2

# ── 데이터 ──
print("[STEP 1] Data")
raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, "2026-01-01", cfg.TECH_INDICATORS)
fg_df = fetch_fear_greed()
fr_df = fetch_funding_rate()
raw = merge_extra_features(raw, fg_df, fr_df)

df_all, fcols = create_features_v2(raw, cfg.TECH_INDICATORS, lookahead=LA, buy_threshold=DZ)
df_all["date"] = pd.to_datetime(df_all["date"])

extra_cols = ["fear_greed", "funding_rate", "funding_rate_ma8",
              "funding_rate_ma24", "funding_rate_cumsum_24h"]
for col in extra_cols:
    if col in df_all.columns:
        fcols.append(col)
fcols = [c for c in fcols if c in df_all.columns]

exp_dir = create_experiment("xgb_visualize", {"la": LA, "dz": DZ})

periods = [
    ("2024-01-01", "2025-01-01", "2024~2025"),
    ("2025-01-01", "2026-01-01", "2025~2026"),
]

for oos_start, oos_end, label in periods:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    oos_mask = (df_all["date"] >= pd.Timestamp(oos_start)) & (df_all["date"] < pd.Timestamp(oos_end))
    oos_indices = df_all[oos_mask].index.tolist()
    oos_df = df_all.loc[oos_indices].reset_index(drop=True)

    # TTA 학습
    all_actions = []
    i = 0
    while i < len(oos_indices):
        chunk_end = min(i + RETRAIN_EVERY, len(oos_indices))
        chunk_indices = oos_indices[i:chunk_end]
        current_time = df_all.loc[chunk_indices[0], "date"]
        train_cutoff = current_time - pd.Timedelta(hours=LA)
        train_df = df_all[df_all["date"] <= train_cutoff].reset_index(drop=True)
        pred_df = df_all.loc[chunk_indices].reset_index(drop=True)

        val_split = int(len(train_df) * (1 - VAL_RATIO))
        X_tr = train_df[fcols].iloc[:val_split]
        y_tr = train_df["target_3class"].iloc[:val_split]
        X_val = train_df[fcols].iloc[val_split:]
        y_val = train_df["target_3class"].iloc[val_split:]

        params = {
            "objective": "multi:softprob", "num_class": 3,
            "eval_metric": "mlogloss", "max_depth": 6,
            "learning_rate": 0.02, "subsample": 0.7,
            "colsample_bytree": 0.7, "min_child_weight": 50,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "tree_method": "hist", "verbosity": 0,
        }
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        dpred = xgb.DMatrix(pred_df[fcols])
        model = xgb.train(params, dtrain, num_boost_round=1000,
                          evals=[(dval, "valid")],
                          early_stopping_rounds=50, verbose_eval=False)
        probs = model.predict(dpred)
        actions = np.argmax(probs, axis=1)
        all_actions.extend(actions.tolist())
        i = chunk_end

    actions_arr = np.array(all_actions)
    result = run_3action(actions_arr, oos_df)

    print(f"  Return={result['model_return_pct']:+.2f}% | B&H={result['buyhold_return_pct']:+.2f}% | "
          f"Sharpe={result['sharpe_ratio']:.2f} | MDD={result['mdd_pct']:.2f}%")

    # ── 시각화 ──
    dates = oos_df["date"].values
    prices = oos_df["close"].values
    pv = np.array(result["portfolio_values"])

    # 매수/매도 타점 추출
    buy_idx = []
    sell_idx = []
    for entry in result["trade_log"]:
        idx, action, price = entry
        if action == "BUY":
            buy_idx.append(idx)
        elif action == "SELL":
            sell_idx.append(idx)

    fig, axes = plt.subplots(2, 1, figsize=(20, 12),
                             gridspec_kw={"height_ratios": [2, 1]},
                             sharex=True)

    # ── 상단: 가격 + 매매 타점 ──
    ax = axes[0]
    ax.plot(dates, prices, color="#333333", linewidth=1, alpha=0.8, label="BTC/USDT")

    if buy_idx:
        ax.scatter(dates[buy_idx], prices[buy_idx],
                   marker="^", color="#2196F3", s=60, zorder=5,
                   label=f"Buy ({len(buy_idx)})", edgecolors="white", linewidth=0.5)
    if sell_idx:
        ax.scatter(dates[sell_idx], prices[sell_idx],
                   marker="v", color="#E53935", s=60, zorder=5,
                   label=f"Sell ({len(sell_idx)})", edgecolors="white", linewidth=0.5)

    # 매수-매도 구간 음영
    i_buy = 0
    i_sell = 0
    trade_log_sorted = sorted(result["trade_log"], key=lambda x: x[0])
    for k in range(len(trade_log_sorted) - 1):
        idx1, act1, _ = trade_log_sorted[k]
        idx2, act2, _ = trade_log_sorted[k + 1]
        if act1 == "BUY" and act2 == "SELL":
            # 수익/손실에 따라 색상
            ret = (prices[idx2] - prices[idx1]) / prices[idx1]
            color = "#2196F3" if ret > 0 else "#E53935"
            ax.axvspan(dates[idx1], dates[idx2], alpha=0.08, color=color)

    ax.set_title(f"XGBoost Base [{label}] | Return={result['model_return_pct']:+.2f}% | "
                 f"B&H={result['buyhold_return_pct']:+.2f}% | Sharpe={result['sharpe_ratio']:.2f} | "
                 f"MDD={result['mdd_pct']:.2f}%",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Price (USDT)", fontsize=11)
    ax.legend(fontsize=10, loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)

    # ── 하단: 포트폴리오 ──
    ax2 = axes[1]
    pv_norm = pv / pv[0]
    bh_norm = prices / prices[0]

    ax2.fill_between(dates[:len(pv_norm)], 1.0, pv_norm,
                     where=pv_norm >= 1.0, alpha=0.3, color="#2196F3", interpolate=True)
    ax2.fill_between(dates[:len(pv_norm)], 1.0, pv_norm,
                     where=pv_norm < 1.0, alpha=0.3, color="#E53935", interpolate=True)
    ax2.plot(dates[:len(pv_norm)], pv_norm, color="#1565C0", linewidth=2,
             label=f"XGB ({result['model_return_pct']:+.2f}%)")
    ax2.plot(dates[:len(bh_norm)], bh_norm, color="#FF9800", linewidth=1.5,
             linestyle="--", alpha=0.7,
             label=f"B&H ({result['buyhold_return_pct']:+.2f}%)")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    ax2.set_ylabel("Portfolio (normalized)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(fontsize=10, loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    sns.despine(ax=ax2, left=True, bottom=True)

    fig.tight_layout()
    path = os.path.join(exp_dir, f"xgb_trades_{label.replace('~','_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Chart] {path}")

print("\nDone!")
