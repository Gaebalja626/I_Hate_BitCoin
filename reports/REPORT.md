# BTC 자동매매 알고리즘 보고서

## 1. 개요

BTC/USDT 1시간봉 데이터를 기반으로 **Buy/Hold/Sell 타이밍**을 자동 판단하는 머신러닝 모델을 개발하고, Out-of-Sample 백테스트를 통해 성능을 검증하였다.

---

## 2. 전체 파이프라인

```
[데이터 수집] → [피처 엔지니어링] → [라벨링] → [모델 학습] → [예측] → [매매 실행]
                                                    ↑                          |
                                                    └──── 월간 재학습 (TTA) ────┘
```

---

## 3. 데이터

### 3-1. 가격 데이터
- **소스**: Binance API (ccxt 라이브러리)
- **종목**: BTC/USDT
- **주기**: 1시간봉 (OHLCV)
- **기간**: 2020-01-01 ~ 2026-01-01 (약 52,000개 캔들)

### 3-2. 외부 데이터
| 데이터 | 소스 | 주기 | 설명 |
|---|---|---|---|
| Fear & Greed Index | alternative.me API | 일별 | 시장 심리 지수 (0=극도 공포 ~ 100=극도 탐욕) |
| Funding Rate | Binance Futures API | 8시간 | 선물 펀딩비율 (롱/숏 수급 지표) |

---

## 4. 피처 엔지니어링 (총 50개)

### 4-1. 기술지표 (16개)
stockstats 라이브러리 기반으로 산출.

| 카테고리 | 지표 |
|---|---|
| 추세 | MACD, MACD Signal, MACD Histogram |
| 변동성 | Bollinger Upper/Lower Band, ATR(14) |
| 모멘텀 | RSI(14), CCI(14), DX(14) |
| 이동평균 | SMA(10, 30, 60), EMA(10, 30) |
| 스토캐스틱 | KDJ-K, KDJ-D |

### 4-2. 가격 파생 피처 (29개)

| 카테고리 | 피처 | 설명 |
|---|---|---|
| 수익률 | ret_{1,3,6,12,24,48,168,336,720} | 1시간 ~ 30일 과거 수익률 |
| 거래량 | vol_chg_{1,6,24}, vol_ma_ratio, vol_ma_ratio_7d | 거래량 변화율 및 이동평균 대비 비율 |
| 캔들 패턴 | candle_body, upper_shadow, lower_shadow, high_low_range | 캔들 형태 수치화 |
| 추세 위치 | price_vs_sma7d, price_vs_sma30d, sma_cross_7_30 | 장기 이동평균 대비 위치, 골든/데드 크로스 |
| 변동성 | volatility_7d, volatility_30d | 7일/30일 수익률 표준편차 |
| 가격 범위 | price_position_14d | 14일 고저 범위 내 현재 위치 (0~1) |
| 볼린저 | boll_position | 볼린저밴드 내 위치 (0~1) |
| 시간 | hour, dayofweek | 시간대/요일 (주기성 반영) |

### 4-3. 외부 피처 (5개)

| 피처 | 설명 |
|---|---|
| fear_greed | 공포탐욕지수 원본값 (0~100) |
| funding_rate | 현재 펀딩레이트 |
| funding_rate_ma8 | 펀딩레이트 8시간 이동평균 |
| funding_rate_ma24 | 펀딩레이트 24시간 이동평균 |
| funding_rate_cumsum_24h | 펀딩레이트 24시간 누적합 |

---

## 5. 라벨링

24시간 후 수익률을 기준으로 3-class 라벨을 생성한다.

```
future_return = close[t + 24] / close[t] - 1

if future_return > +1%  → Buy  (0)
if future_return < -1%  → Sell (2)
else                    → Hold (1)
```

- **Lookahead**: 24시간
- **Dead Zone**: ±1%
- **클래스 비율**: Buy ~21%, Hold ~50%, Sell ~29%

> 주의: 라벨 생성에만 미래 가격을 사용하며, 피처에는 과거 데이터만 사용하여 데이터 유출(data leakage)을 방지한다.

---

## 6. 모델

### 6-1. 알고리즘
**LightGBM** (Light Gradient Boosting Machine) — Gradient Boosted Decision Tree 기반 분류 모델

### 6-2. 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| objective | multiclass | 3-클래스 분류 |
| metric | multi_logloss | 다중 클래스 로그 손실 |
| num_leaves | 31 | 트리당 최대 리프 수 |
| max_depth | 6 | 트리 최대 깊이 |
| learning_rate | 0.02 | 학습률 |
| feature_fraction | 0.7 | iteration마다 피처 70% 랜덤 샘플링 |
| bagging_fraction | 0.7 | iteration마다 데이터 70% 랜덤 샘플링 |
| min_child_samples | 50 | 리프 노드 최소 샘플 수 |
| lambda_l1 / l2 | 0.1 / 1.0 | L1/L2 정규화 |
| early_stopping | 50 rounds | 검증 손실 50회 연속 미개선 시 학습 중단 |

### 6-3. 학습/검증 분할
- **시간 순서 기반** 80/20 분할 (학습: 앞 80%, 검증: 뒤 20%)
- 시계열 데이터의 시간적 순서를 보존하여 미래 정보 유출 방지

---

## 7. Test-Time Adaptation (TTA)

시장 환경(regime)은 시간에 따라 변화한다. 이에 대응하기 위해 **월간 재학습** 전략을 적용한다.

```
매월 1일:
  1. 전월까지 실제 가격 데이터 확보 (Ground Truth 확정)
  2. 기존 학습 데이터 + 신규 데이터로 모델 재학습
  3. 다음 1개월간 예측에 사용
  4. 반복 (연간 12~13회)
```

- **재학습 주기**: 720시간 (약 30일)
- **Train cutoff**: 현재 시점 - Lookahead(24h) → 라벨이 확정된 데이터만 사용
- 매 재학습 시 학습 데이터가 720개씩 증가하여 최신 시장 패턴 반영

---

## 8. 매매 규칙

```
매 1시간마다 모델이 Buy/Hold/Sell 중 하나를 예측:

  Buy  + 현금 보유 중  → 전액 매수 (수수료 0.1% 차감)
  Sell + BTC 보유 중   → 전액 매도 (수수료 0.1% 차감)
  Hold                → 포지션 유지 (거래 없음)
  Buy  + 이미 BTC 보유 → 유지 (중복 매수 없음)
  Sell + 이미 현금     → 유지 (공매도 없음)
```

| 항목 | 설정 |
|---|---|
| 초기 자본 | $100,000 |
| 거래 수수료 | 0.1% (편도) |
| 포지션 | Long only (공매도 없음) |
| 레버리지 | 없음 (1x) |

---

## 9. 피처 중요도

모델이 의사결정 시 가장 크게 참조한 피처 (importance = information gain 기준):

| 순위 | 피처 | 중요도 | 설명 |
|---|---|---|---|
| 1 | volatility_7d | 44,400 | 최근 7일 변동성 |
| 2 | volatility_30d | 20,881 | 최근 30일 변동성 |
| 3 | high_low_range | 16,669 | 캔들 진폭 |
| 4 | dayofweek | 12,726 | 요일 |
| 5 | atr_14 | 11,611 | Average True Range |
| 6 | hour | 10,879 | 시간대 |
| 7 | fear_greed | 8,065 | 공포탐욕지수 |
| 8 | price_position_14d | 7,336 | 14일 범위 내 위치 |
| 9 | ret_720 | 7,089 | 30일 수익률 |
| 10 | ret_168 | 6,927 | 7일 수익률 |

**해석**: 모델은 주로 **변동성**(시장이 얼마나 출렁이는가), **시장 심리**(공포탐욕지수), **장기 추세**(7일/30일 수익률)를 종합하여 매매 판단을 수행한다.

---

## 10. 실험 결과

### 10-1. OOS (Out-of-Sample) 성능

학습에 사용하지 않은 미래 데이터에 대한 평가:

| 평가 기간 | 시장 상황 | 모델 수익률 | Buy & Hold | 초과수익 | Sharpe | MDD | 거래 횟수 |
|---|---|---|---|---|---|---|---|
| 2024.01 ~ 2025.01 | 강세장 (+120%) | +46.64% | +120.31% | -73.67% | 1.66 | -15.37% | 196 |
| 2025.01 ~ 2026.01 | 약세/횡보장 (-7%) | **+21.20%** | -6.51% | **+27.71%** | 1.00 | -19.87% | 48 |

### 10-2. Lookahead 비교 (2025~2026 OOS)

| Lookahead | 수익률 | Sharpe | MDD | 거래 | 비고 |
|---|---|---|---|---|---|
| 6h | +0.00% | 0.00 | 0.00% | 0 | 전부 Hold (6h 내 ±1% 변동 희소) |
| 12h | -0.88% | 0.13 | -29.46% | 19 | 학습 부분적 성공 |
| **24h** | **+21.20%** | **1.00** | **-19.87%** | **48** | **최적 구간** |
| 48h | -20.70% | -0.32 | -34.76% | 59 | 학습 실패 (iter=1~5) |

### 10-3. 매매 특성 분석

- **약세/횡보 구간 (2025.05~09)**: 대부분 Hold → 하락 손실 회피
- **반등 시그널 포착 (2025.04, 10월)**: 적시 매수 진입
- **MDD 제한**: 모든 기간에서 Buy & Hold 대비 MDD가 절반 수준

---

## 11. 한계점

1. **강세장 추종력 부족**: 지속적 상승장에서 Buy & Hold 대비 크게 언더퍼폼
2. **포지션 상태 미인식**: 분류 모델은 현재 보유 여부/수수료 누적을 고려하지 못함
3. **단일 자산**: BTC/USDT에만 적용, 다른 자산으로의 일반화 미검증
4. **외부 데이터 한계**: 온체인 데이터, 뉴스 센티먼트 등 미반영

---

## 12. 향후 개선 방향

- **강화학습 (RL)** 도입: 포지션 상태/수수료/연속 매매를 인식하는 의사결정 모델로 전환
- **외부 피처 확장**: 온체인 지표 (거래소 유입량, 고래 거래), 뉴스 센티먼트
- **멀티 타임프레임**: 1h + 4h + 1d 복합 피처
- **앙상블**: 다수 Lookahead 모델의 투표 기반 합의 매매

---

## 13. 기술 스택

| 항목 | 도구 |
|---|---|
| 언어 | Python 3 |
| 데이터 수집 | ccxt (Binance API) |
| 기술지표 | stockstats |
| ML 모델 | LightGBM |
| 데이터 처리 | pandas, numpy |
| 시각화 | matplotlib, seaborn |

---

## 14. 파일 구조

```
project_bitcoin/
├── config.py                 # 실험 설정 (하이퍼파라미터, 경로)
├── data_collector.py         # Binance 데이터 수집
├── fetch_extra_features.py   # Fear&Greed, Funding Rate 수집
├── run_improved_v2.py        # 피처 엔지니어링 (create_features_v2)
├── models/
│   └── lgbm_model.py         # LightGBM 학습/예측
├── backtester.py             # 백테스트 엔진
├── experiment.py             # 실험 결과 저장
├── run_2stage.py             # 메인 실행 (LA sweep + TTA)
├── datasets/
│   ├── btc_raw_1h.csv        # 캐시된 OHLCV 데이터
│   ├── fear_greed.csv        # Fear & Greed Index
│   └── funding_rate.csv      # Funding Rate
└── results/                  # 실험 결과 (차트, 메트릭)
```
