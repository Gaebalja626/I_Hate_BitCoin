"""
XGBoost Optuna 튜닝 (50 trials)
- CPU 부하 제한: n_jobs=2, thread 제한
- 2025~2026 OOS 수익률 기준 최적화
- LA=24h, DZ=1%, TTA
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.environ["SSL_CERT_FILE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"
os.environ["REQUESTS_CA_BUNDLE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"
os.environ["OMP_NUM_THREADS"] = "2"

import xgboost as xgb
import config as cfg
from data_collector import load_or_fetch
from backtester import run_3action
from feature_engineer import create_features_v2
from fetch_extra_features import fetch_fear_greed, fetch_funding_rate, merge_extra_features
from experiment import create_experiment

LA = 24; DZ = 0.01; RETRAIN_EVERY = 720; VAL_RATIO = 0.2

# ── 데이터 (1회만 로드) ──
print("[데이터 로드]")
raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, "2026-01-01", cfg.TECH_INDICATORS)
fg_df = fetch_fear_greed(); fr_df = fetch_funding_rate()
raw = merge_extra_features(raw, fg_df, fr_df)
df_all, fcols = create_features_v2(raw, cfg.TECH_INDICATORS, lookahead=LA, buy_threshold=DZ)
df_all["date"] = pd.to_datetime(df_all["date"])
extra_cols = ["fear_greed", "funding_rate", "funding_rate_ma8", "funding_rate_ma24", "funding_rate_cumsum_24h"]
for col in extra_cols:
    if col in df_all.columns: fcols.append(col)
fcols = [c for c in fcols if c in df_all.columns]

# OOS 인덱스 미리 계산
oos_mask = (df_all["date"] >= pd.Timestamp("2025-01-01")) & (df_all["date"] < pd.Timestamp("2026-01-01"))
oos_indices = df_all[oos_mask].index.tolist()
oos_df = df_all.loc[oos_indices].reset_index(drop=True)

# 청크 미리 계산
chunks = []
i = 0
while i < len(oos_indices):
    chunk_end = min(i + RETRAIN_EVERY, len(oos_indices))
    chunk_indices = oos_indices[i:chunk_end]
    current_time = df_all.loc[chunk_indices[0], "date"]
    train_cutoff = current_time - pd.Timedelta(hours=LA)
    train_df = df_all[df_all["date"] <= train_cutoff].reset_index(drop=True)
    pred_df = df_all.loc[chunk_indices].reset_index(drop=True)

    val_split = int(len(train_df) * (1 - VAL_RATIO))
    dtrain = xgb.DMatrix(train_df[fcols].iloc[:val_split], label=train_df["target_3class"].iloc[:val_split])
    dval = xgb.DMatrix(train_df[fcols].iloc[val_split:], label=train_df["target_3class"].iloc[val_split:])
    dpred = xgb.DMatrix(pred_df[fcols])
    chunks.append((dtrain, dval, dpred))
    i = chunk_end

print(f"[준비 완료] {len(chunks)} chunks, {len(oos_indices)} OOS samples")

exp_dir = create_experiment("xgb_optuna", {"trials": 50, "la": LA, "dz": DZ})

best_results = []


def objective(trial):
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "verbosity": 0,
        "nthread": 2,
        # 탐색 범위
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 5.0),
    }
    num_boost = trial.suggest_int("num_boost_round", 100, 1500)

    all_actions = []
    for dtrain, dval, dpred in chunks:
        model = xgb.train(params, dtrain, num_boost_round=num_boost,
                          evals=[(dval, "valid")],
                          early_stopping_rounds=50, verbose_eval=False)
        probs = model.predict(dpred)
        actions = np.argmax(probs, axis=1)
        all_actions.extend(actions.tolist())

    actions_arr = np.array(all_actions)
    result = run_3action(actions_arr, oos_df)

    # Sharpe를 주 목표, 수익률을 보조
    sharpe = result["sharpe_ratio"]
    ret = result["model_return_pct"]
    mdd = result["mdd_pct"]
    trades = result["total_trades"]

    # 거래가 너무 적으면 페널티
    if trades < 10:
        return -10.0

    trial.set_user_attr("return", ret)
    trial.set_user_attr("sharpe", sharpe)
    trial.set_user_attr("mdd", mdd)
    trial.set_user_attr("trades", trades)

    return sharpe  # maximize


study = optuna.create_study(direction="maximize", study_name="xgb_tuning")

print(f"\n[Optuna] 50 trials 시작 (CPU 제한: 2 threads)")
print(f"{'Trial':>6} | {'Sharpe':>7} | {'Return':>8} | {'MDD':>8} | {'Trades':>6} | {'Time':>6}")
print("-" * 60)

t_start = time.time()

def callback(study, trial):
    ret = trial.user_attrs.get("return", 0)
    sharpe = trial.user_attrs.get("sharpe", 0)
    mdd = trial.user_attrs.get("mdd", 0)
    trades = trial.user_attrs.get("trades", 0)
    elapsed = time.time() - t_start

    is_best = trial.number == study.best_trial.number
    marker = " *BEST*" if is_best else ""
    print(f"  {trial.number:>4d} | {sharpe:>+6.2f} | {ret:>+7.2f}% | {mdd:>7.2f}% | {trades:>6d} | "
          f"{elapsed:>5.0f}s{marker}")

study.optimize(objective, n_trials=50, callbacks=[callback])

# ── 결과 정리 ──
best = study.best_trial
print(f"\n{'='*70}")
print(f"  Best Trial #{best.number}")
print(f"  Sharpe: {best.user_attrs['sharpe']:.2f}")
print(f"  Return: {best.user_attrs['return']:+.2f}%")
print(f"  MDD:    {best.user_attrs['mdd']:.2f}%")
print(f"  Trades: {best.user_attrs['trades']}")
print(f"{'='*70}")
print(f"\n  Best Params:")
for k, v in best.params.items():
    print(f"    {k}: {v}")

# 기존 base와 비교
print(f"\n  비교:")
print(f"    XGB base:  Return=+5.77%, Sharpe=0.39, MDD=-15.67%")
print(f"    XGB tuned: Return={best.user_attrs['return']:+.2f}%, "
      f"Sharpe={best.user_attrs['sharpe']:.2f}, MDD={best.user_attrs['mdd']:.2f}%")

# 저장
summary = {
    "best_trial": best.number,
    "best_sharpe": best.user_attrs["sharpe"],
    "best_return": best.user_attrs["return"],
    "best_mdd": best.user_attrs["mdd"],
    "best_trades": best.user_attrs["trades"],
    "best_params": best.params,
    "all_trials": [
        {
            "number": t.number,
            "sharpe": t.user_attrs.get("sharpe", 0),
            "return": t.user_attrs.get("return", 0),
            "mdd": t.user_attrs.get("mdd", 0),
            "trades": t.user_attrs.get("trades", 0),
            "params": t.params,
        }
        for t in study.trials
    ],
}

with open(os.path.join(exp_dir, "optuna_results.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

# ── 2024~2025에서도 best params로 실행 ──
print(f"\n[Best Params로 2024~2025 검증]")
oos_mask_24 = (df_all["date"] >= pd.Timestamp("2024-01-01")) & (df_all["date"] < pd.Timestamp("2025-01-01"))
oos_indices_24 = df_all[oos_mask_24].index.tolist()
oos_df_24 = df_all.loc[oos_indices_24].reset_index(drop=True)

best_params = {
    "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
    "tree_method": "hist", "verbosity": 0, "nthread": 2,
}
best_params.update({k: v for k, v in best.params.items() if k != "num_boost_round"})
best_num_boost = best.params["num_boost_round"]

all_actions_24 = []
i = 0
while i < len(oos_indices_24):
    chunk_end = min(i + RETRAIN_EVERY, len(oos_indices_24))
    chunk_indices = oos_indices_24[i:chunk_end]
    current_time = df_all.loc[chunk_indices[0], "date"]
    train_cutoff = current_time - pd.Timedelta(hours=LA)
    train_df = df_all[df_all["date"] <= train_cutoff].reset_index(drop=True)
    pred_df = df_all.loc[chunk_indices].reset_index(drop=True)
    val_split = int(len(train_df) * (1 - VAL_RATIO))
    dtrain = xgb.DMatrix(train_df[fcols].iloc[:val_split], label=train_df["target_3class"].iloc[:val_split])
    dval = xgb.DMatrix(train_df[fcols].iloc[val_split:], label=train_df["target_3class"].iloc[val_split:])
    dpred = xgb.DMatrix(pred_df[fcols])
    model = xgb.train(best_params, dtrain, num_boost_round=best_num_boost,
                      evals=[(dval, "valid")], early_stopping_rounds=50, verbose_eval=False)
    probs = model.predict(dpred)
    all_actions_24.extend(np.argmax(probs, axis=1).tolist())
    i = chunk_end

result_24 = run_3action(np.array(all_actions_24), oos_df_24)
print(f"  2024~2025: Return={result_24['model_return_pct']:+.2f}%, "
      f"Sharpe={result_24['sharpe_ratio']:.2f}, MDD={result_24['mdd_pct']:.2f}%")

print(f"\n  최종 비교:")
print(f"  {'':>12s} | {'2024~2025':>12s} | {'2025~2026':>12s}")
print(f"  {'XGB base':>12s} | {'Return=+14.99%':>12s} | {'Return=+5.77%':>12s}")
print(f"  {'XGB optuna':>12s} | Return={result_24['model_return_pct']:>+.2f}% | "
      f"Return={best.user_attrs['return']:>+.2f}%")

print(f"\n[Saved] {exp_dir}")
