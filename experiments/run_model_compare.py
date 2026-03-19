"""
모델 비교 실험: LightGBM (기본 / 튜닝) vs XGBoost vs CatBoost
- LA=24h, DZ=1%, TTA(월간 재학습)
- Fear&Greed 1일 shift 적용
- OOS: 2025-01-01 ~ 2026-01-01
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

os.environ["SSL_CERT_FILE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"
os.environ["REQUESTS_CA_BUNDLE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

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

# ── 모델 정의 ──

def train_predict_lgbm_base(train_df, pred_df, fcols):
    """LightGBM 기본 (현재 설정)"""
    params = cfg.LGBM_PARAMS_3CLASS.copy()
    val_split = int(len(train_df) * (1 - VAL_RATIO))
    X_tr, y_tr = train_df[fcols].iloc[:val_split], train_df["target_3class"].iloc[:val_split]
    X_val, y_val = train_df[fcols].iloc[val_split:], train_df["target_3class"].iloc[val_split:]

    ds_tr = lgb.Dataset(X_tr, label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    model = lgb.train(params, ds_tr, num_boost_round=1000, valid_sets=[ds_val],
                      valid_names=["valid"],
                      callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)])
    probs = model.predict(pred_df[fcols])
    actions = np.argmax(probs, axis=1)
    return actions, model.best_iteration


def train_predict_lgbm_tuned(train_df, pred_df, fcols):
    """LightGBM 튜닝 (더 깊은 트리, 낮은 LR, 더 많은 잎)"""
    params = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.01,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.6,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "lambda_l1": 0.05,
        "lambda_l2": 0.5,
        "min_gain_to_split": 0.01,
        "path_smooth": 0.1,
        "verbose": -1, "n_jobs": -1,
    }
    val_split = int(len(train_df) * (1 - VAL_RATIO))
    X_tr, y_tr = train_df[fcols].iloc[:val_split], train_df["target_3class"].iloc[:val_split]
    X_val, y_val = train_df[fcols].iloc[val_split:], train_df["target_3class"].iloc[val_split:]

    ds_tr = lgb.Dataset(X_tr, label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    model = lgb.train(params, ds_tr, num_boost_round=2000, valid_sets=[ds_val],
                      valid_names=["valid"],
                      callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100)])
    probs = model.predict(pred_df[fcols])
    actions = np.argmax(probs, axis=1)
    return actions, model.best_iteration


def train_predict_xgboost(train_df, pred_df, fcols):
    """XGBoost"""
    params = {
        "objective": "multi:softprob", "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "learning_rate": 0.02,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "verbosity": 0,
    }
    val_split = int(len(train_df) * (1 - VAL_RATIO))
    X_tr, y_tr = train_df[fcols].iloc[:val_split], train_df["target_3class"].iloc[:val_split]
    X_val, y_val = train_df[fcols].iloc[val_split:], train_df["target_3class"].iloc[val_split:]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dpred = xgb.DMatrix(pred_df[fcols])

    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=50, verbose_eval=False)
    probs = model.predict(dpred)
    actions = np.argmax(probs, axis=1)
    return actions, model.best_iteration


def train_predict_xgboost_tuned(train_df, pred_df, fcols):
    """XGBoost 튜닝"""
    params = {
        "objective": "multi:softprob", "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 8,
        "learning_rate": 0.01,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "min_child_weight": 30,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "gamma": 0.1,
        "tree_method": "hist",
        "verbosity": 0,
    }
    val_split = int(len(train_df) * (1 - VAL_RATIO))
    X_tr, y_tr = train_df[fcols].iloc[:val_split], train_df["target_3class"].iloc[:val_split]
    X_val, y_val = train_df[fcols].iloc[val_split:], train_df["target_3class"].iloc[val_split:]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dpred = xgb.DMatrix(pred_df[fcols])

    model = xgb.train(params, dtrain, num_boost_round=2000,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=100, verbose_eval=False)
    probs = model.predict(dpred)
    actions = np.argmax(probs, axis=1)
    return actions, model.best_iteration


def train_predict_catboost(train_df, pred_df, fcols):
    """CatBoost 기본"""
    val_split = int(len(train_df) * (1 - VAL_RATIO))
    X_tr, y_tr = train_df[fcols].iloc[:val_split], train_df["target_3class"].iloc[:val_split]
    X_val, y_val = train_df[fcols].iloc[val_split:], train_df["target_3class"].iloc[val_split:]

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.02,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        bagging_temperature=0.7,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        early_stopping_rounds=50,
        verbose=0,
        thread_count=-1,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    probs = model.predict_proba(pred_df[fcols])
    actions = np.argmax(probs, axis=1)
    return actions, model.best_iteration_


def train_predict_catboost_tuned(train_df, pred_df, fcols):
    """CatBoost 튜닝"""
    val_split = int(len(train_df) * (1 - VAL_RATIO))
    X_tr, y_tr = train_df[fcols].iloc[:val_split], train_df["target_3class"].iloc[:val_split]
    X_val, y_val = train_df[fcols].iloc[val_split:], train_df["target_3class"].iloc[val_split:]

    model = CatBoostClassifier(
        iterations=2000,
        depth=8,
        learning_rate=0.01,
        l2_leaf_reg=1.0,
        random_strength=0.5,
        bagging_temperature=0.5,
        border_count=128,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        early_stopping_rounds=100,
        verbose=0,
        thread_count=-1,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    probs = model.predict_proba(pred_df[fcols])
    actions = np.argmax(probs, axis=1)
    return actions, model.best_iteration_


MODELS = {
    "LGBM_base":   train_predict_lgbm_base,
    "LGBM_tuned":  train_predict_lgbm_tuned,
    "XGB_base":    train_predict_xgboost,
    "XGB_tuned":   train_predict_xgboost_tuned,
    "CatB_base":   train_predict_catboost,
    "CatB_tuned":  train_predict_catboost_tuned,
}

# ── 데이터 ──
print("[STEP 1] Data + Extra Features")
raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, "2026-01-01", cfg.TECH_INDICATORS)
fg_df = fetch_fear_greed()
fr_df = fetch_funding_rate()
raw = merge_extra_features(raw, fg_df, fr_df)

print(f"[STEP 2] Features v2 (LA={LA}h, DZ={DZ*100:.0f}%)")
df_all, fcols = create_features_v2(raw, cfg.TECH_INDICATORS, lookahead=LA, buy_threshold=DZ)
df_all["date"] = pd.to_datetime(df_all["date"])

extra_cols = ["fear_greed", "funding_rate", "funding_rate_ma8",
              "funding_rate_ma24", "funding_rate_cumsum_24h"]
for col in extra_cols:
    if col in df_all.columns:
        fcols.append(col)
fcols = [c for c in fcols if c in df_all.columns]

# ── OOS 기간 ──
periods = [
    ("2024-01-01", "2025-01-01", "2024~2025"),
    ("2025-01-01", "2026-01-01", "2025~2026"),
]

exp_dir = create_experiment("model_compare", {
    "models": list(MODELS.keys()),
    "la": LA, "dz": DZ, "retrain_every": RETRAIN_EVERY,
})

all_results = []

for oos_start, oos_end, period_label in periods:
    print(f"\n{'='*70}")
    print(f"  OOS: {period_label}")
    print(f"{'='*70}")

    oos_mask = (df_all["date"] >= pd.Timestamp(oos_start)) & (df_all["date"] < pd.Timestamp(oos_end))
    oos_indices = df_all[oos_mask].index.tolist()
    oos_df = df_all.loc[oos_indices].reset_index(drop=True)

    period_results = {}

    for model_name, train_fn in MODELS.items():
        print(f"\n  --- {model_name} (TTA) ---")
        t0 = time.time()
        all_actions = []

        i = 0
        retrain_count = 0
        while i < len(oos_indices):
            chunk_end = min(i + RETRAIN_EVERY, len(oos_indices))
            chunk_indices = oos_indices[i:chunk_end]

            current_time = df_all.loc[chunk_indices[0], "date"]
            train_cutoff = current_time - pd.Timedelta(hours=LA)
            train_df = df_all[df_all["date"] <= train_cutoff].reset_index(drop=True)
            pred_df = df_all.loc[chunk_indices].reset_index(drop=True)

            retrain_count += 1
            try:
                actions, best_iter = train_fn(train_df, pred_df, fcols)
                all_actions.extend(actions.tolist())
                nb = int(sum(actions == 0))
                nh = int(sum(actions == 1))
                ns = int(sum(actions == 2))
                print(f"    [#{retrain_count}] {current_time.strftime('%Y-%m-%d')} | "
                      f"iter={best_iter} | B={nb} H={nh} S={ns}")
            except Exception as e:
                print(f"    [#{retrain_count}] ERROR: {e}")
                # fallback: all hold
                all_actions.extend([1] * len(chunk_indices))

            i = chunk_end

        elapsed = time.time() - t0
        actions_arr = np.array(all_actions)
        result = run_3action(actions_arr, oos_df)

        nb_total = int(sum(actions_arr == 0))
        nh_total = int(sum(actions_arr == 1))
        ns_total = int(sum(actions_arr == 2))

        print(f"  [{model_name}] Return={result['model_return_pct']:+.2f}% | "
              f"B&H={result['buyhold_return_pct']:+.2f}% | "
              f"Sharpe={result['sharpe_ratio']:.2f} | MDD={result['mdd_pct']:.2f}% | "
              f"Trades={result['total_trades']} | {elapsed:.1f}s")

        period_results[model_name] = result
        all_results.append({
            "period": period_label,
            "model": model_name,
            "return": result["model_return_pct"],
            "bh": result["buyhold_return_pct"],
            "sharpe": result["sharpe_ratio"],
            "mdd": result["mdd_pct"],
            "trades": result["total_trades"],
            "buy": nb_total, "hold": nh_total, "sell": ns_total,
            "elapsed_sec": round(elapsed, 1),
        })

    # ── 비교 차트 ──
    fig, ax = plt.subplots(figsize=(18, 7))
    colors = ["#2196F3", "#1565C0", "#E53935", "#B71C1C", "#4CAF50", "#2E7D32"]

    for idx, (model_name, result) in enumerate(period_results.items()):
        pv = np.array(result["portfolio_values"])
        pv_norm = pv / pv[0]
        dates = oos_df["date"].values[:len(pv_norm)]
        ax.plot(dates, pv_norm, color=colors[idx], linewidth=2,
                label=f"{model_name} ({result['model_return_pct']:+.2f}%)")

    # B&H
    prices = oos_df["close"].values
    bh_norm = prices / prices[0]
    bh_ret = (prices[-1] / prices[0] - 1) * 100
    ax.plot(oos_df["date"].values[:len(bh_norm)], bh_norm,
            color="#FF9800", linewidth=2, linestyle="--", alpha=0.7,
            label=f"B&H ({bh_ret:+.2f}%)")

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(f"Model Compare [{period_label}] | LA={LA}h DZ={DZ*100:.0f}% + TTA",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio (normalized)")
    ax.legend(fontsize=10, loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()

    path = os.path.join(exp_dir, f"compare_{period_label.replace('~','_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Chart] {path}")

# ── 최종 요약 ──
print(f"\n{'='*100}")
print(f"{'Period':>12} | {'Model':>14} | {'Return':>9} | {'B&H':>8} | "
      f"{'Sharpe':>7} | {'MDD':>8} | {'Trades':>6} | {'Time':>6}")
print("-" * 100)
for r in all_results:
    print(f"  {r['period']:>10} | {r['model']:>14} | {r['return']:>+8.2f}% | {r['bh']:>+7.2f}% | "
          f"{r['sharpe']:>6.2f} | {r['mdd']:>7.2f}% | {r['trades']:>6} | {r['elapsed_sec']:>5.1f}s")
print(f"{'='*100}")

with open(os.path.join(exp_dir, "summary.json"), "w") as f:
    json.dump({"results": all_results}, f, indent=2)
print(f"\n[Saved] {exp_dir}")
