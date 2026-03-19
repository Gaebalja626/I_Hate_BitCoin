"""
Bitcoin RL Trading - FinRL 기반
바이낸스(ccxt)에서 BTC/USDT 데이터 수집 → 보조지표 추가 → PPO 학습
"""
import os
import sys
import calendar
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. 바이낸스 데이터 수집 (ccxt)
# ============================================================

def fetch_btc_data(
    pair="BTC/USDT",
    timeframe="1h",
    start_date="2023-01-01",
    end_date="2025-01-01",
):
    """바이낸스에서 OHLCV 데이터를 가져옵니다."""
    exchange = ccxt.binance({"enableRateLimit": True})

    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end_date}T00:00:00Z")

    all_ohlcv = []
    limit = 1000

    print(f"[데이터 수집] {pair} {timeframe} | {start_date} ~ {end_date}")

    while since < end_ts:
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # 마지막 타임스탬프 다음부터
        print(f"  수집중... {datetime.utcfromtimestamp(since/1000).strftime('%Y-%m-%d %H:%M')}", end="\r")

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[df["timestamp"] <= end_ts].copy()
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(np.float64)

    print(f"\n[완료] 총 {len(df)}개 캔들 수집 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    return df


# ============================================================
# 2. 보조지표 추가
# ============================================================

TECH_INDICATORS = [
    "macd",          # MACD
    "boll_ub",       # 볼린저밴드 상단
    "boll_lb",       # 볼린저밴드 하단
    "rsi_30",        # RSI (30)
    "cci_30",        # CCI (30)
    "dx_30",         # DX (30)
    "close_30_sma",  # 30기간 이동평균
    "close_60_sma",  # 60기간 이동평균
]

def add_technical_indicators(df, indicator_list=None):
    """stockstats로 보조지표를 추가합니다."""
    if indicator_list is None:
        indicator_list = TECH_INDICATORS

    temp = df[["date", "open", "high", "low", "close", "volume"]].copy()
    stock_df = Sdf.retype(temp)

    for indicator in indicator_list:
        try:
            df[indicator] = stock_df[indicator].values
        except Exception as e:
            print(f"  [경고] {indicator} 계산 실패: {e}")
            df[indicator] = 0.0

    # NaN/Inf 제거
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    print(f"[보조지표] {len(indicator_list)}개 추가 완료 → 최종 {len(df)}행")
    return df


# ============================================================
# 3. Gymnasium 환경 (BTC 단일 종목)
# ============================================================

class BtcTradingEnv(gym.Env):
    """BTC 트레이딩 강화학습 환경 (Gymnasium 호환)"""
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        initial_balance=100_000,
        trading_fee=0.001,      # 0.1% 수수료
        tech_indicator_list=None,
        reward_scaling=1e-4,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list or TECH_INDICATORS

        # state: [balance, price, holdings, tech_indicators...]
        self.state_dim = 3 + len(self.tech_indicator_list)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        # action: [-1, 1] → 매도~매수
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.max_step = len(self.df) - 1
        self.reset()

    def _get_state(self):
        row = self.df.iloc[self.day]
        tech_values = [row[ind] for ind in self.tech_indicator_list]
        state = np.array(
            [
                self.balance / self.initial_balance,     # 정규화된 잔고
                row["close"] / self.df["close"].iloc[0], # 정규화된 가격
                self.holdings,                            # BTC 보유량
            ] + tech_values,
            dtype=np.float32,
        )
        # 보조지표 정규화 (간단히 스케일링)
        state[3:] = state[3:] / (np.abs(state[3:]).max() + 1e-8)
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.balance = self.initial_balance
        self.holdings = 0.0
        self.total_asset = self.initial_balance
        self.trades = 0
        self.asset_history = [self.initial_balance]
        return self._get_state(), {}

    def step(self, action):
        action_val = float(action[0])
        price = self.df.iloc[self.day]["close"]

        # 매수
        if action_val > 0:
            buy_amount = action_val * self.balance / price
            cost = buy_amount * price * (1 + self.trading_fee)
            if cost <= self.balance:
                self.holdings += buy_amount
                self.balance -= cost
                self.trades += 1
        # 매도
        elif action_val < 0:
            sell_amount = min(abs(action_val) * self.holdings, self.holdings)
            if sell_amount > 0:
                self.balance += sell_amount * price * (1 - self.trading_fee)
                self.holdings -= sell_amount
                self.trades += 1

        # 다음 스텝
        self.day += 1
        new_price = self.df.iloc[self.day]["close"]
        new_total = self.balance + self.holdings * new_price

        reward = (new_total - self.total_asset) * self.reward_scaling
        self.total_asset = new_total
        self.asset_history.append(new_total)

        terminated = self.day >= self.max_step
        truncated = False

        return self._get_state(), reward, terminated, truncated, {}


# ============================================================
# 4. 학습 & 백테스트
# ============================================================

def train_ppo(train_df, tech_indicators, total_timesteps=50_000):
    """PPO 모델 학습"""
    env = DummyVecEnv([lambda: BtcTradingEnv(train_df, tech_indicator_list=tech_indicators)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        verbose=1,
    )

    print(f"\n{'='*60}")
    print(f"[학습 시작] PPO | timesteps={total_timesteps}")
    print(f"{'='*60}")
    model.learn(total_timesteps=total_timesteps)

    os.makedirs("trained_models", exist_ok=True)
    model.save("trained_models/ppo_btc")
    print("[학습 완료] 모델 저장: trained_models/ppo_btc.zip")
    return model


def backtest(model, test_df, tech_indicators):
    """백테스트 실행 & 결과 시각화"""
    env = BtcTradingEnv(test_df, tech_indicator_list=tech_indicators)
    state, _ = env.reset()

    while True:
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # 수익률 계산
    final_asset = env.total_asset
    initial = env.initial_balance
    ret = (final_asset - initial) / initial * 100

    # Buy & Hold 비교
    bh_ret = (test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1) * 100

    print(f"\n{'='*60}")
    print(f"[백테스트 결과]")
    print(f"  초기 자산:   ${initial:,.0f}")
    print(f"  최종 자산:   ${final_asset:,.0f}")
    print(f"  PPO 수익률:  {ret:+.2f}%")
    print(f"  B&H 수익률:  {bh_ret:+.2f}%")
    print(f"  총 거래 횟수: {env.trades}")
    print(f"{'='*60}")

    # 차트 저장
    os.makedirs("results", exist_ok=True)

    asset_norm = np.array(env.asset_history) / initial
    bh_prices = test_df["close"].values / test_df["close"].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.plot(asset_norm, label=f"PPO Agent ({ret:+.2f}%)", linewidth=2)
    plt.plot(bh_prices, label=f"Buy & Hold ({bh_ret:+.2f}%)", linewidth=2, alpha=0.7)
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    plt.title("BTC/USDT Trading - PPO vs Buy & Hold")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value (normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/btc_ppo_backtest.png", dpi=150)
    print("[차트 저장] results/btc_ppo_backtest.png")

    return env.asset_history


# ============================================================
# 5. 메인 실행
# ============================================================

if __name__ == "__main__":
    # --- 설정 ---
    PAIR = "BTC/USDT"
    TIMEFRAME = "1h"                    # 1시간봉
    TRAIN_START = "2023-01-01"
    TRAIN_END = "2024-06-01"
    TEST_START = "2024-06-01"
    TEST_END = "2025-01-01"
    TOTAL_TIMESTEPS = 50_000            # 간단 학습 (늘리면 성능 향상)

    DATA_DIR = "datasets"
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- 1) 데이터 수집 ---
    train_csv = f"{DATA_DIR}/btc_train.csv"
    test_csv = f"{DATA_DIR}/btc_test.csv"

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print("[캐시] 기존 데이터 로드")
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
    else:
        raw = fetch_btc_data(PAIR, TIMEFRAME, TRAIN_START, TEST_END)
        raw = add_technical_indicators(raw, TECH_INDICATORS)

        # train / test 분할
        split_date = pd.Timestamp(TEST_START)
        raw["date"] = pd.to_datetime(raw["date"])
        train_df = raw[raw["date"] < split_date].reset_index(drop=True)
        test_df = raw[raw["date"] >= split_date].reset_index(drop=True)

        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        print(f"[저장] {train_csv} ({len(train_df)}행), {test_csv} ({len(test_df)}행)")

    print(f"\n학습 데이터: {len(train_df)}행 | 테스트 데이터: {len(test_df)}행")
    print(f"보조지표: {TECH_INDICATORS}")

    # --- 2) 학습 ---
    model = train_ppo(train_df, TECH_INDICATORS, total_timesteps=TOTAL_TIMESTEPS)

    # --- 3) 백테스트 ---
    backtest(model, test_df, TECH_INDICATORS)
