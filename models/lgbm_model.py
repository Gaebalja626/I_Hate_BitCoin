"""
LightGBM 모델 모듈
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    LGBM_PARAMS, LGBM_PARAMS_3CLASS, LGBM_PARAMS_REG,
    LGBM_NUM_BOOST, LGBM_EARLY_STOP, LGBM_VAL_RATIO,
)


def train(train_df, feature_cols, params=None):
    """
    LightGBM 학습 (시계열 기반 validation split + early stopping).

    Returns:
        model: 학습된 LightGBM Booster
        feat_imp: 피처 중요도 DataFrame
        info: 학습 메타정보 dict
    """
    if params is None:
        params = LGBM_PARAMS

    val_split = int(len(train_df) * (1 - LGBM_VAL_RATIO))
    X_train = train_df[feature_cols].iloc[:val_split]
    y_train = train_df["target"].iloc[:val_split]
    X_val = train_df[feature_cols].iloc[val_split:]
    y_val = train_df["target"].iloc[val_split:]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    print(f"[학습] LightGBM | train={len(X_train)} | val={len(X_val)}")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=LGBM_NUM_BOOST,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.log_evaluation(100),
            lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOP),
        ],
    )

    # 피처 중요도
    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    info = {
        "best_iteration": model.best_iteration,
        "best_score": model.best_score.get("valid", {}).get("binary_logloss", None),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "num_features": len(feature_cols),
    }

    print(f"[학습 완료] best_iter={info['best_iteration']} | val_loss={info['best_score']:.6f}")
    return model, feat_imp, info


def predict(model, df, feature_cols):
    """예측 확률을 반환합니다."""
    X = df[feature_cols]
    return model.predict(X)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    print(f"[모델 저장] {path}")


def load_model(path):
    return lgb.Booster(model_file=path)


# ============================================================
# 3-class (Buy/Hold/Sell) 모델
# ============================================================
ACTION_NAMES = {0: "Buy", 1: "Hold", 2: "Sell"}


def train_3class(train_df, feature_cols, params=None):
    """
    LightGBM 3-class 학습 (Buy=0, Hold=1, Sell=2).

    Returns:
        model, feat_imp, info
    """
    if params is None:
        params = LGBM_PARAMS_3CLASS

    target_col = "target_3class"

    val_split = int(len(train_df) * (1 - LGBM_VAL_RATIO))
    X_train = train_df[feature_cols].iloc[:val_split]
    y_train = train_df[target_col].iloc[:val_split]
    X_val = train_df[feature_cols].iloc[val_split:]
    y_val = train_df[target_col].iloc[val_split:]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 클래스 분포 출력
    for cls in [0, 1, 2]:
        cnt = (y_train == cls).sum()
        print(f"  {ACTION_NAMES[cls]}: {cnt} ({cnt/len(y_train):.1%})")

    print(f"[학습] LightGBM 3-class | train={len(X_train)} | val={len(X_val)}")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=LGBM_NUM_BOOST,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.log_evaluation(100),
            lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOP),
        ],
    )

    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    best_score = model.best_score.get("valid", {}).get("multi_logloss", None)
    info = {
        "best_iteration": model.best_iteration,
        "best_score": best_score,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "num_features": len(feature_cols),
    }
    print(f"[학습 완료] best_iter={info['best_iteration']} | val_loss={best_score:.6f}")
    return model, feat_imp, info


def predict_3class(model, df, feature_cols):
    """
    3-class 예측 → 액션 인덱스 + 확률 반환.

    Returns:
        actions: 0=Buy, 1=Hold, 2=Sell 배열
        probs: (N, 3) 확률 배열
    """
    X = df[feature_cols]
    probs = model.predict(X)  # shape (N, 3)
    actions = np.argmax(probs, axis=1)
    return actions, probs


# ============================================================
# Regression 모델 (수익률 예측)
# ============================================================

def train_regression(train_df, feature_cols, params=None):
    """
    LightGBM regression: future_ret (48h 뒤 수익률) 예측.

    Returns:
        model, feat_imp, info
    """
    if params is None:
        params = LGBM_PARAMS_REG

    target_col = "future_ret"

    val_split = int(len(train_df) * (1 - LGBM_VAL_RATIO))
    X_train = train_df[feature_cols].iloc[:val_split]
    y_train = train_df[target_col].iloc[:val_split]
    X_val = train_df[feature_cols].iloc[val_split:]
    y_val = train_df[target_col].iloc[val_split:]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    print(f"[학습] LightGBM Regression | train={len(X_train)} | val={len(X_val)}")
    print(f"  target mean={y_train.mean():.6f} std={y_train.std():.6f}")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=LGBM_NUM_BOOST,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.log_evaluation(100),
            lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOP),
        ],
    )

    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    best_score = model.best_score.get("valid", {}).get("l1", None)
    info = {
        "best_iteration": model.best_iteration,
        "best_score": best_score,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "num_features": len(feature_cols),
    }
    print(f"[학습 완료] best_iter={info['best_iteration']} | val_mae={best_score:.6f}")
    return model, feat_imp, info


def predict_regression(model, df, feature_cols):
    """수익률 예측값 반환."""
    X = df[feature_cols]
    return model.predict(X)
