# I_Hate_BitCoin

BTC/USDT 1시간봉 기반 3-class 분류 트레이딩 시스템.
GBT 모델(LightGBM, XGBoost, CatBoost)이 매 시점마다 Buy/Hold/Sell 행동을 결정하고, TTA(Test-Time Adaptation)로 월별 재학습한다.

## Quick Start

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/Gaebalja626/I_Hate_BitCoin.git
cd I_Hate_BitCoin

# (권장) conda 환경 생성
conda create -n btc python=3.10 -y
conda activate btc

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 수집

첫 실행 시 자동으로 `datasets/` 폴더에 OHLCV 데이터를 다운로드한다.
인터넷 연결이 필요하며, ccxt를 통해 Binance에서 가져온다.

### 3. 실험 실행

```bash
# 6개 모델 비교 (LGBM/XGB/CatBoost × base/tuned)
python experiments/run_model_compare.py

# XGBoost Optuna 튜닝 (50 trials)
python experiments/run_xgb_optuna.py

# 앙상블 (top 3 모델)
python experiments/run_ensemble.py

# TTA ablation study
python experiments/run_ablation_tta.py

# XGBoost 매매 시각화
python experiments/run_xgb_visualize.py

# XGBoost 행동 분석
python experiments/run_xgb_behavior_analysis.py
```

결과는 `results/` 폴더에 실험별 디렉토리로 저장된다.

## 프로젝트 구조

```
├── config.py                # 하이퍼파라미터, 데이터 설정
├── data_collector.py        # Binance OHLCV 수집 (ccxt)
├── feature_engineer.py      # 기술지표 + 파생 피처 생성
├── fetch_extra_features.py  # Fear & Greed, Funding Rate
├── backtester.py            # 3-action 백테스터 (수수료 포함)
├── experiment.py            # 실험 디렉토리 생성/관리
├── models/
│   └── lgbm_model.py        # LightGBM 학습/예측
├── experiments/
│   ├── run_model_compare.py       # 6모델 비교
│   ├── run_xgb_optuna.py         # Optuna 튜닝
│   ├── run_ensemble.py            # 앙상블
│   ├── run_ablation_tta.py        # TTA ablation
│   ├── run_xgb_visualize.py       # 매매 시각화
│   ├── run_xgb_behavior_analysis.py # 행동 분석
│   └── ...
├── reports/                 # 분석 보고서 (.md)
├── datasets/                # OHLCV 캐시 (gitignore)
├── results/                 # 실험 결과 (gitignore)
└── trained_models/          # 저장된 모델 (gitignore)
```

## 핵심 개념

| 용어 | 설명 |
|------|------|
| **3-class** | Buy(0) / Hold(1) / Sell(2) — 모델이 매 시간 행동 결정 |
| **LA (Lookahead)** | 라벨 생성 시 미래 참조 기간 (기본 24시간) |
| **DZ (Dead Zone)** | 수익률 ±DZ 이내면 Hold로 분류 (기본 1%) |
| **TTA** | 720시간(≈30일)마다 expanding window로 재학습 |
| **OOS** | Out-of-Sample — 2024\~2025, 2025\~2026 구간 |

## 주요 결과

- **LGBM base가 최고 성능**: 2024\~2025 +58.68%, 2025\~2026 −1.51%
- **TTA 필수**: TTA 없으면 2025\~2026에서 −22% → TTA 있으면 −1.5%
- **튜닝 역설**: 복잡한 모델일수록 과적합 → base 파라미터가 최적
- **앙상블 한계**: 같은 피처/라벨의 GBT 모델은 다양성 부족으로 앙상블 효과 미미

## 참고

- Python 3.10+ 권장
- SSL 인증서 오류 시 `certifi` 패키지 경로를 config에서 수정
- Binance API 접근이 차단된 환경에서는 VPN 필요
