"""
앙상블 실험: Top 3 모델 다수결 투표
- LGBM_base, XGB_base, CatB_tuned
- 투표 방식: (1) Hard voting (다수결) (2) Soft voting (확률 평균)
- LA=24h, DZ=1%, TTA(월간 재학습)
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from collections import Counter

os.environ["SSL_CERT_FILE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"
os.environ["REQUESTS_CA_BUNDLE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

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


def train_predict_lgbm_base(train_df, pred_df, fcols):
    params = cfg.LGBM_PARAMS_3CLASS.copy()
    val_split = int(len(train_df) * (1 - VAL_RATIO))
    X_tr, y_tr = train_df[fcols].iloc[:val_split], train_df["target_3class"].iloc[:val_split]
    X_val, y_val = train_df[fcols].iloc[val_split:], train_df["target_3class"].iloc[val_split:]
    ds_tr = lgb.Dataset(X_tr, label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    model = lgb.train(params, ds_tr, num_boost_round=1000, valid_sets=[ds_val],
                      valid_names=["valid"],
                      callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)])
    probs = model.predict(pred_df[fcols])  # (N, 3)
    return np.argmax(probs, axis=1), probs, model.best_iteration


def train_predict_xgb_base(train_df, pred_df, fcols):
    params = {
        "objective": "multi:softprob", "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 6, "learning_rate": 0.02,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "min_child_weight": 50,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "tree_method": "hist", "verbosity": 0,
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
    probs = model.predict(dpred)  # (N, 3)
    return np.argmax(probs, axis=1), probs, model.best_iteration


def train_predict_catb_tuned(train_df, pred_df, fcols):
    val_split = int(len(train_df) * (1 - VAL_RATIO))
    X_tr, y_tr = train_df[fcols].iloc[:val_split], train_df["target_3class"].iloc[:val_split]
    X_val, y_val = train_df[fcols].iloc[val_split:], train_df["target_3class"].iloc[val_split:]
    model = CatBoostClassifier(
        iterations=2000, depth=8, learning_rate=0.01,
        l2_leaf_reg=1.0, random_strength=0.5, bagging_temperature=0.5,
        border_count=128,
        loss_function="MultiClass", eval_metric="MultiClass",
        early_stopping_rounds=100, verbose=0, thread_count=-1,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    probs = model.predict_proba(pred_df[fcols])  # (N, 3)
    return np.argmax(probs, axis=1), probs, model.best_iteration_


MODEL_FNS = {
    "LGBM_base": train_predict_lgbm_base,
    "XGB_base":  train_predict_xgb_base,
    "CatB_tuned": train_predict_catb_tuned,
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

# ── OOS ──
periods = [
    ("2024-01-01", "2025-01-01", "2024~2025"),
    ("2025-01-01", "2026-01-01", "2025~2026"),
]

exp_dir = create_experiment("ensemble_top3", {
    "models": list(MODEL_FNS.keys()),
    "methods": ["hard_voting", "soft_voting", "individual"],
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

    # 각 모델별 actions + probs 수집
    model_all_actions = {name: [] for name in MODEL_FNS}
    model_all_probs = {name: [] for name in MODEL_FNS}

    t0 = time.time()
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
        print(f"\n  [#{retrain_count}] {current_time.strftime('%Y-%m-%d')} | chunk={len(chunk_indices)}")

        for name, fn in MODEL_FNS.items():
            actions, probs, best_iter = fn(train_df, pred_df, fcols)
            model_all_actions[name].extend(actions.tolist())
            model_all_probs[name].append(probs)
            nb = int(sum(actions == 0))
            nh = int(sum(actions == 1))
            ns = int(sum(actions == 2))
            print(f"    {name:>12s} | iter={best_iter:>4d} | B={nb} H={nh} S={ns}")

        i = chunk_end

    elapsed = time.time() - t0
    print(f"\n  Total training time: {elapsed:.1f}s")

    # probs 합치기
    for name in MODEL_FNS:
        model_all_probs[name] = np.vstack(model_all_probs[name])  # (N, 3)

    N = len(oos_indices)

    # ── (1) 개별 모델 결과 ──
    individual_results = {}
    for name in MODEL_FNS:
        actions_arr = np.array(model_all_actions[name])
        result = run_3action(actions_arr, oos_df)
        individual_results[name] = result
        all_results.append({
            "period": period_label, "model": name,
            "return": result["model_return_pct"], "bh": result["buyhold_return_pct"],
            "sharpe": result["sharpe_ratio"], "mdd": result["mdd_pct"],
            "trades": result["total_trades"],
        })
        print(f"  [{name}] Return={result['model_return_pct']:+.2f}% | "
              f"Sharpe={result['sharpe_ratio']:.2f} | MDD={result['mdd_pct']:.2f}%")

    # ── (2) Hard Voting (다수결) ──
    hard_actions = np.zeros(N, dtype=int)
    for j in range(N):
        votes = [model_all_actions[name][j] for name in MODEL_FNS]
        hard_actions[j] = Counter(votes).most_common(1)[0][0]

    result_hard = run_3action(hard_actions, oos_df)
    all_results.append({
        "period": period_label, "model": "Ensemble_Hard",
        "return": result_hard["model_return_pct"], "bh": result_hard["buyhold_return_pct"],
        "sharpe": result_hard["sharpe_ratio"], "mdd": result_hard["mdd_pct"],
        "trades": result_hard["total_trades"],
    })
    nb = int(sum(hard_actions == 0))
    nh = int(sum(hard_actions == 1))
    ns = int(sum(hard_actions == 2))
    print(f"  [Hard Vote] Return={result_hard['model_return_pct']:+.2f}% | "
          f"Sharpe={result_hard['sharpe_ratio']:.2f} | MDD={result_hard['mdd_pct']:.2f}% | "
          f"B={nb} H={nh} S={ns}")

    # ── (3) Soft Voting (확률 평균) ──
    avg_probs = np.mean([model_all_probs[name] for name in MODEL_FNS], axis=0)  # (N, 3)
    soft_actions = np.argmax(avg_probs, axis=1)

    result_soft = run_3action(soft_actions, oos_df)
    all_results.append({
        "period": period_label, "model": "Ensemble_Soft",
        "return": result_soft["model_return_pct"], "bh": result_soft["buyhold_return_pct"],
        "sharpe": result_soft["sharpe_ratio"], "mdd": result_soft["mdd_pct"],
        "trades": result_soft["total_trades"],
    })
    nb = int(sum(soft_actions == 0))
    nh = int(sum(soft_actions == 1))
    ns = int(sum(soft_actions == 2))
    print(f"  [Soft Vote] Return={result_soft['model_return_pct']:+.2f}% | "
          f"Sharpe={result_soft['sharpe_ratio']:.2f} | MDD={result_soft['mdd_pct']:.2f}% | "
          f"B={nb} H={nh} S={ns}")

    # ── (4) 보수적 앙상블: Buy는 만장일치, Sell은 2/3 이상 ──
    conservative_actions = np.ones(N, dtype=int)  # 기본 Hold
    for j in range(N):
        votes = [model_all_actions[name][j] for name in MODEL_FNS]
        buy_count = votes.count(0)
        sell_count = votes.count(2)
        if buy_count == 3:  # 만장일치 Buy
            conservative_actions[j] = 0
        elif sell_count >= 2:  # 2/3 이상 Sell
            conservative_actions[j] = 2

    result_cons = run_3action(conservative_actions, oos_df)
    all_results.append({
        "period": period_label, "model": "Ensemble_Conservative",
        "return": result_cons["model_return_pct"], "bh": result_cons["buyhold_return_pct"],
        "sharpe": result_cons["sharpe_ratio"], "mdd": result_cons["mdd_pct"],
        "trades": result_cons["total_trades"],
    })
    nb = int(sum(conservative_actions == 0))
    nh = int(sum(conservative_actions == 1))
    ns = int(sum(conservative_actions == 2))
    print(f"  [Conservative] Return={result_cons['model_return_pct']:+.2f}% | "
          f"Sharpe={result_cons['sharpe_ratio']:.2f} | MDD={result_cons['mdd_pct']:.2f}% | "
          f"B={nb} H={nh} S={ns}")

    # ── 비교 차트 ──
    fig, ax = plt.subplots(figsize=(18, 7))
    all_plot = {
        **{name: r for name, r in individual_results.items()},
        "Ensemble_Hard": result_hard,
        "Ensemble_Soft": result_soft,
        "Ensemble_Conservative": result_cons,
    }
    colors = ["#2196F3", "#E53935", "#4CAF50", "#9C27B0", "#FF5722", "#00BCD4", "#795548"]

    for idx, (name, result) in enumerate(all_plot.items()):
        pv = np.array(result["portfolio_values"])
        pv_norm = pv / pv[0]
        dates = oos_df["date"].values[:len(pv_norm)]
        lw = 3 if "Ensemble" in name else 1.5
        ls = "-" if "Ensemble" in name else "--"
        ax.plot(dates, pv_norm, color=colors[idx], linewidth=lw, linestyle=ls,
                label=f"{name} ({result['model_return_pct']:+.2f}%)")

    # B&H
    prices = oos_df["close"].values
    bh_norm = prices / prices[0]
    bh_ret = (prices[-1] / prices[0] - 1) * 100
    ax.plot(oos_df["date"].values[:len(bh_norm)], bh_norm,
            color="#FF9800", linewidth=2, linestyle=":", alpha=0.7,
            label=f"B&H ({bh_ret:+.2f}%)")

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(f"Ensemble Top3 [{period_label}] | LA={LA}h DZ={DZ*100:.0f}% + TTA",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio (normalized)")
    ax.legend(fontsize=9, loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()

    path = os.path.join(exp_dir, f"ensemble_{period_label.replace('~','_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Chart] {path}")

# ── 최종 요약 ──
print(f"\n{'='*95}")
print(f"{'Period':>12} | {'Model':>22} | {'Return':>9} | {'B&H':>8} | "
      f"{'Sharpe':>7} | {'MDD':>8} | {'Trades':>6}")
print("-" * 95)
for r in all_results:
    print(f"  {r['period']:>10} | {r['model']:>22} | {r['return']:>+8.2f}% | {r['bh']:>+7.2f}% | "
          f"{r['sharpe']:>6.2f} | {r['mdd']:>7.2f}% | {r['trades']:>6}")
print(f"{'='*95}")

with open(os.path.join(exp_dir, "summary.json"), "w") as f:
    json.dump({"results": all_results}, f, indent=2)
print(f"\n[Saved] {exp_dir}")
