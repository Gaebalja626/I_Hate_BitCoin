"""
백테스트 엔진
- 시그널 기반 Long/Flat 전략
"""
import numpy as np
from sklearn.metrics import accuracy_score

from config import INITIAL_BALANCE, TRADING_FEE


def run(predictions, test_df, threshold=0.5,
        initial_balance=None, trading_fee=None):
    """
    백테스트 실행.

    Args:
        predictions: 상승 확률 배열
        test_df: 테스트 DataFrame (close 컬럼 필요)
        threshold: 매수 기준 확률

    Returns:
        result dict: 수익률, 거래횟수, 포트폴리오 히스토리 등
    """
    if initial_balance is None:
        initial_balance = INITIAL_BALANCE
    if trading_fee is None:
        trading_fee = TRADING_FEE

    prices = test_df["close"].values
    signals = (predictions > threshold).astype(int)

    balance = float(initial_balance)
    holdings = 0.0
    total_trades = 0
    portfolio = [initial_balance]

    for i in range(len(prices) - 1):
        price = prices[i]
        if signals[i] == 1 and holdings == 0:
            holdings = balance / price * (1 - trading_fee)
            balance = 0
            total_trades += 1
        elif signals[i] == 0 and holdings > 0:
            balance = holdings * price * (1 - trading_fee)
            holdings = 0
            total_trades += 1

        total_asset = balance + holdings * prices[i + 1]
        portfolio.append(total_asset)

    if holdings > 0:
        balance = holdings * prices[-1] * (1 - trading_fee)
        holdings = 0
        portfolio[-1] = balance

    final_asset = portfolio[-1]
    model_ret = (final_asset - initial_balance) / initial_balance * 100
    bh_ret = (prices[-1] / prices[0] - 1) * 100

    # 정확도 (타겟이 있을 때만)
    acc = None
    if "target" in test_df.columns:
        y_true = test_df["target"].values
        preds = (predictions > threshold).astype(int)
        acc = accuracy_score(y_true, preds)

    # 최대 낙폭 (MDD)
    pv = np.array(portfolio)
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak
    mdd = drawdown.min() * 100

    # 샤프 비율 (연율화, 무위험 0%)
    returns = np.diff(pv) / pv[:-1]
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24)  # 1h 기준
    else:
        sharpe = 0.0

    result = {
        "threshold": threshold,
        "initial_balance": initial_balance,
        "final_asset": final_asset,
        "model_return_pct": round(model_ret, 2),
        "buyhold_return_pct": round(bh_ret, 2),
        "total_trades": total_trades,
        "accuracy": round(acc, 4) if acc is not None else None,
        "signal_ratio": round(signals.mean(), 4),
        "mdd_pct": round(mdd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "portfolio_values": portfolio,
        "predictions": predictions.tolist(),
    }
    return result


def run_3action(actions, test_df, initial_balance=None, trading_fee=None):
    """
    3-action 백테스트 (Buy=0, Hold=1, Sell=2).

    - Buy: 현금 → 매수 (이미 보유 중이면 유지)
    - Hold: 현재 포지션 유지 (거래 없음)
    - Sell: 보유 → 매도 (이미 현금이면 유지)
    """
    if initial_balance is None:
        initial_balance = INITIAL_BALANCE
    if trading_fee is None:
        trading_fee = TRADING_FEE

    prices = test_df["close"].values
    balance = float(initial_balance)
    holdings = 0.0
    total_trades = 0
    portfolio = [initial_balance]
    trade_log = []  # (시점, 액션, 가격)

    for i in range(len(prices) - 1):
        price = prices[i]
        action = actions[i]  # 0=Buy, 1=Hold, 2=Sell

        if action == 0 and holdings == 0:  # Buy
            holdings = balance / price * (1 - trading_fee)
            balance = 0
            total_trades += 1
            trade_log.append((i, "BUY", price))
        elif action == 2 and holdings > 0:  # Sell
            balance = holdings * price * (1 - trading_fee)
            holdings = 0
            total_trades += 1
            trade_log.append((i, "SELL", price))
        # action == 1 (Hold) → 아무것도 안 함

        total_asset = balance + holdings * prices[i + 1]
        portfolio.append(total_asset)

    # 마지막 정리
    if holdings > 0:
        balance = holdings * prices[-1] * (1 - trading_fee)
        holdings = 0
        portfolio[-1] = balance

    final_asset = portfolio[-1]
    model_ret = (final_asset - initial_balance) / initial_balance * 100
    bh_ret = (prices[-1] / prices[0] - 1) * 100

    # 정확도 (3-class)
    acc = None
    if "target_3class" in test_df.columns:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(test_df["target_3class"].values, actions)

    # MDD
    pv = np.array(portfolio)
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak
    mdd = drawdown.min() * 100

    # Sharpe
    returns = np.diff(pv) / pv[:-1]
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(365 * 24)

    # 액션 분포
    action_counts = {0: 0, 1: 0, 2: 0}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1

    result = {
        "threshold": "3-action",
        "initial_balance": initial_balance,
        "final_asset": final_asset,
        "model_return_pct": round(model_ret, 2),
        "buyhold_return_pct": round(bh_ret, 2),
        "total_trades": total_trades,
        "accuracy": round(acc, 4) if acc is not None else None,
        "signal_ratio": round(action_counts[0] / len(actions), 4),
        "mdd_pct": round(mdd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "portfolio_values": portfolio,
        "predictions": actions.tolist() if hasattr(actions, 'tolist') else list(actions),
        "action_distribution": {
            "buy": action_counts[0],
            "hold": action_counts[1],
            "sell": action_counts[2],
        },
        "trade_log": trade_log,
    }
    return result


def print_result(result):
    """백테스트 결과를 출력합니다."""
    print(f"\n{'='*60}")
    print(f"[백테스트 결과] threshold={result['threshold']}")
    print(f"  초기 자산:      ${result['initial_balance']:,.0f}")
    print(f"  최종 자산:      ${result['final_asset']:,.0f}")
    print(f"  모델 수익률:    {result['model_return_pct']:+.2f}%")
    print(f"  Buy&Hold:       {result['buyhold_return_pct']:+.2f}%")
    print(f"  MDD:            {result['mdd_pct']:.2f}%")
    print(f"  Sharpe Ratio:   {result['sharpe_ratio']:.2f}")
    print(f"  정확도:         {result['accuracy']}")
    print(f"  매수 시그널:    {result['signal_ratio']:.2%}")
    print(f"  총 거래:        {result['total_trades']}회")
    print(f"{'='*60}")
