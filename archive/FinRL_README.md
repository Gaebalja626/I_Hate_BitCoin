# FinRL 레포지토리 레퍼런스

> 강화학습 기반 금융 트레이딩 프레임워크. Train → Test → Trade 파이프라인.

## 프로젝트 구조

```
FinRL/
├── finrl/
│   ├── config.py                # 날짜범위, 하이퍼파라미터, API키
│   ├── config_tickers.py        # 티커 목록 (DOW30, NASDAQ100, S&P500 등)
│   ├── train.py / test.py / trade.py  # 파이프라인 진입점
│   ├── main.py                  # CLI (--mode=train/test/trade)
│   ├── meta/
│   │   ├── data_processor.py    # 통합 데이터 처리 래퍼
│   │   ├── data_processors/     # 개별 데이터 소스 프로세서
│   │   │   ├── processor_yahoofinance.py
│   │   │   ├── processor_alpaca.py
│   │   │   ├── processor_wrds.py
│   │   │   ├── processor_ccxt.py
│   │   │   ├── processor_binance.py
│   │   │   ├── processor_eodhd.py
│   │   │   ├── processor_joinquant.py
│   │   │   └── processor_sinopac.py
│   │   ├── preprocessor/
│   │   │   ├── preprocessors.py       # FeatureEngineer, GroupByScaler
│   │   │   ├── yahoodownloader.py     # 간단 Yahoo 다운로더
│   │   │   └── tusharedownloader.py
│   │   └── env_stock_trading/
│   │       ├── env_stocktrading.py       # SB3 호환 환경
│   │       ├── env_stocktrading_np.py    # ElegantRL 호환 환경
│   │       ├── env_stocktrading_cashpenalty.py
│   │       └── env_stocktrading_stoploss.py
│   ├── agents/
│   │   ├── stablebaselines3/models.py   # A2C, PPO, DDPG, SAC, TD3
│   │   ├── elegantrl/models.py
│   │   └── rllib/models.py
│   └── applications/
│       ├── stock_trading/
│       │   ├── stock_trading.py              # 기본 워크플로우
│       │   └── stock_trading_rolling_window.py
│       └── cryptocurrency_trading/
└── examples/                    # Jupyter 노트북 튜토리얼
```

---

## 데이터 소스

| 소스 | 시장 | 최소 간격 | API 키 | 비고 |
|------|------|-----------|--------|------|
| **YahooFinance** | US 주식 | 1min | 불필요 | 가장 쉬움, 무료 |
| **Alpaca** | US 주식/ETF | 1min | 필요 | Paper trading 지원 |
| **WRDS** | US 전체 | 1ms (tick) | 필요 | 학술 계정 |
| **Binance/CCXT** | 암호화폐 | 1min | 선택 | CCXT로 다수 거래소 지원 |
| **Tushare** | 중국 A주 | 1min | 필요 | |
| **JoinQuant** | 중국 | 1min | 필요 | |
| **Baostock** | 중국 | 5min | 불필요 | 1990년~ |
| **Akshare** | 중국 | 1day | 불필요 | |
| **EOD Historical** | 글로벌 50+국가 | 1day | 필요 | |
| **Sinopac** | 대만 | 1min | 필요 | |

---

## 데이터 파이프라인

```
다운로드(OHLCV) → 클린(결측치) → 기술지표 추가 → VIX/Turbulence → numpy array
```

### 핵심 API: DataProcessor

```python
from finrl.meta.data_processor import DataProcessor

dp = DataProcessor(data_source="yahoofinance")  # or "alpaca", "wrds"

# 1. 다운로드
data = dp.download_data(
    ticker_list=["AAPL", "MSFT", "GOOG"],
    start_date="2020-01-01",
    end_date="2024-01-01",
    time_interval="1D"  # 1m, 5m, 15m, 30m, 1h, 1D
)

# 2. 클린
data = dp.clean_data(data)

# 3. 기술지표
data = dp.add_technical_indicator(data, [
    "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
    "dx_30", "close_30_sma", "close_60_sma"
])

# 4. VIX (선택)
data = dp.add_vix(data)

# 5. numpy 변환
price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix=True)
```

### DataFrame 출력 포맷

```
date | tic | open | high | low | close | volume | macd | boll_ub | boll_lb | rsi_30 | cci_30 | dx_30 | close_30_sma | close_60_sma | vix | turbulence
```

---

## 사전 정의된 티커 목록

```python
from finrl.config_tickers import (
    DOW_30_TICKER,     # 미국 다우존스 30
    NAS_100_TICKER,    # 나스닥 100
    SP_500_TICKER,     # S&P 500
    HSI_50_TICKER,     # 홍콩 항셍 50
    SSE_50_TICKER,     # 상하이 50
    CSI_300_TICKER,    # 중국 CSI 300
    CAC_40_TICKER,     # 프랑스 CAC 40
    DAX_30_TICKER,     # 독일 DAX 30
    LQ45_TICKER,       # 인도네시아 LQ45
    TAI_0050_TICKER,   # 대만 50
    FX_TICKER,         # 환율
)
```

---

## 환경 (Gym)

### StockTradingEnv (SB3 호환)

```python
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

env_kwargs = {
    "hmax": 100,                          # 종목당 최대 거래 주수
    "initial_amount": 1_000_000,          # 초기 자본금
    "num_stock_shares": [0] * stock_dim,  # 초기 보유량
    "buy_cost_pct": [0.001] * stock_dim,  # 매수 수수료 0.1%
    "sell_cost_pct": [0.001] * stock_dim, # 매도 수수료 0.1%
    "state_space": state_dim,
    "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dim,
    "reward_scaling": 1e-4,
}

env = StockTradingEnv(df=train_data, **env_kwargs)
```

**State 구조**: `[cash, prices(n), holdings(n), tech_indicators(n*m)]`
- state_dim = 1 + 2*n + n*m (n=종목수, m=지표수)

**Action**: 연속값 [-1, 1] × stock_dim
- -1: 최대 매도, 0: 보유, +1: 최대 매수
- 실제 주수 = action × hmax

### StockTradingEnv (ElegantRL 호환, numpy 기반)

```python
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

env = StockTradingEnv(config={
    "price_array": price_array,
    "tech_array": tech_array,
    "turbulence_array": turbulence_array,
    "if_train": True,
})
```

**State 구조**: `[cash, turbulence, turb_bool, prices(n), stocks(n), cooldown(n), tech(m)]`
- state_dim = 1 + 2 + 3*n + m

---

## 에이전트 & 학습

### Stable-Baselines3

```python
from finrl.agents.stablebaselines3.models import DRLAgent

agent = DRLAgent(env=env_train)

# 모델 선택: "a2c", "ppo", "ddpg", "sac", "td3"
model = agent.get_model("ppo", model_kwargs={
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
})

trained = agent.train_model(model, tb_log_name="ppo", total_timesteps=100000)
trained.save("./trained_models/ppo")
```

### 추론/테스트

```python
account_memory, actions_memory = DRLAgent.DRL_prediction(
    model=trained, environment=env_test, deterministic=True
)
```

### 기본 하이퍼파라미터

| 모델 | lr | batch | buffer | 기타 |
|------|-----|-------|--------|------|
| A2C | 7e-4 | - | - | n_steps=5, ent_coef=0.01 |
| PPO | 2.5e-4 | 64 | - | n_steps=2048, ent_coef=0.01 |
| DDPG | 1e-3 | 128 | 50K | |
| TD3 | 1e-3 | 100 | 1M | |
| SAC | 1e-4 | 64 | 100K | ent_coef=auto_0.1 |

---

## 전체 파이프라인 (train.py)

```python
from finrl.train import train

train(
    start_date="2020-01-01",
    end_date="2023-12-31",
    ticker_list=DOW_30_TICKER,
    data_source="yahoofinance",
    time_interval="1D",
    technical_indicator_list=INDICATORS,
    drl_lib="stable_baselines3",  # or "elegantrl", "rllib"
    env=StockTradingEnv,
    model_name="ppo",
    if_vix=True,
    cwd="./trained_models/ppo",
    total_timesteps=100000,
)
```

---

## 설정 기본값 (config.py)

```python
TRAIN_START_DATE = "2014-01-06"
TRAIN_END_DATE   = "2020-07-31"
TEST_START_DATE  = "2020-08-01"
TEST_END_DATE    = "2021-10-01"
TRADE_START_DATE = "2021-11-01"
TRADE_END_DATE   = "2021-12-01"

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
RESULTS_DIR = "results"

INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
              "dx_30", "close_30_sma", "close_60_sma"]
```

---

## 주요 의존성

```
numpy, pandas, gymnasium, yfinance, stockstats
stable-baselines3[extra], elegantrl, ray[tune]
alpaca-py, ccxt, wrds
matplotlib, pyfolio-reloaded, tensorboard
```

---

## Feature Engineering (FeatureEngineer)

```python
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False,
)
processed = fe.preprocess_data(df)
```

**기술지표** (stockstats 기반):
- MACD, 볼린저밴드 (상단/하단), RSI(30), CCI(30), DX(30), SMA(30/60)

**Turbulence Index**:
- 252일 롤링 공분산 기반 시장 불안정도
- 공식: (수익률 - 평균)ᵀ × Σ⁻¹ × (수익률 - 평균)

---

## 출력 파일

- `datasets/train.csv`, `datasets/test.csv` — 전처리된 데이터
- `trained_models/{model}.zip` (SB3) 또는 `act.pth` (ElegantRL)
- `results/account_value_{mode}_{model}.csv` — 일별 포트폴리오 가치
- `results/actions_{mode}_{model}.csv` — 일별 거래 내역
