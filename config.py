"""
실험 설정 모듈
- 데이터, 모델, 백테스트 관련 모든 설정을 한 곳에서 관리
"""

# ── 데이터 ──
PAIR = "BTC/USDT"
TIMEFRAME = "1h"
TRAIN_START = "2020-01-01"
TEST_START = "2024-06-01"
DATA_END = "2025-01-01"

# ── 피처 ──
LOOKAHEAD = 6  # N시간 후 상승/하락 예측

TECH_INDICATORS = [
    "macd", "macds", "macdh",
    "boll_ub", "boll_lb",
    "rsi_14", "cci_14", "dx_14",
    "close_10_sma", "close_30_sma", "close_60_sma",
    "close_10_ema", "close_30_ema",
    "atr_14",
    "kdjk", "kdjd",
]

# ── LightGBM ──
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

LGBM_PARAMS_3CLASS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}
LGBM_PARAMS_REG = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}
LGBM_NUM_BOOST = 1000
LGBM_EARLY_STOP = 50
LGBM_VAL_RATIO = 0.2  # 학습 데이터 중 검증용 비율

# ── 백테스트 ──
INITIAL_BALANCE = 100_000
TRADING_FEE = 0.001       # 0.1%
THRESHOLDS = [0.40, 0.45, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70]

# ── 3-Action 설정 ──
BUY_THRESHOLD = 0.005     # 미래 수익률 > 0.5% → Buy
ACTION_THRESHOLDS = [0.003, 0.005, 0.007, 0.01]  # dead zone 스윕용

# ── 경로 ──
DATA_DIR = "datasets"
RESULTS_DIR = "results"
MODEL_DIR = "trained_models"
