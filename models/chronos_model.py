"""
Chronos 시계열 예측 모델 모듈
- Amazon Chronos T5 기반 확률적 시계열 예측
- 가격 예측 → 트레이딩 시그널 변환
"""
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline


def load_model(model_name="amazon/chronos-t5-tiny", device="cpu"):
    """Chronos 모델 로드."""
    print(f"[Chronos] 모델 로딩: {model_name} (device={device})")
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )
    print(f"[Chronos] 로딩 완료")
    return pipeline


def predict_rolling(pipeline, prices, context_len=512, horizon=6,
                    num_samples=20, step=1):
    """
    롤링 윈도우 방식으로 예측을 수행합니다.

    Args:
        pipeline: Chronos 파이프라인
        prices: close 가격 배열
        context_len: 모델에 넣을 과거 데이터 길이
        horizon: 예측할 미래 스텝 수
        num_samples: 확률 샘플 수
        step: 몇 스텝마다 예측할지 (1=매 스텝)

    Returns:
        predictions: 각 시점의 horizon 스텝 후 예측 중앙값 배열
        pred_upper: 80% 상위 예측
        pred_lower: 80% 하위 예측
    """
    n = len(prices)
    predictions = np.full(n, np.nan)
    pred_upper = np.full(n, np.nan)
    pred_lower = np.full(n, np.nan)

    total_steps = (n - context_len - horizon) // step
    print(f"[Chronos] 롤링 예측 시작 | context={context_len} | horizon={horizon} | 총 {total_steps}회")

    for i in range(context_len, n - horizon, step):
        context = torch.tensor(prices[i - context_len:i], dtype=torch.float32)

        forecast = pipeline.predict(
            context.unsqueeze(0),
            horizon,
            num_samples=num_samples,
        )  # shape: (1, num_samples, horizon)

        samples = forecast[0].numpy()  # (num_samples, horizon)
        # horizon 스텝 후의 가격 예측
        final_step = samples[:, -1]
        predictions[i] = np.median(final_step)
        pred_upper[i] = np.percentile(final_step, 90)
        pred_lower[i] = np.percentile(final_step, 10)

        if (i - context_len) % 200 == 0:
            pct = (i - context_len) / (n - context_len - horizon) * 100
            print(f"  진행: {pct:.1f}% ({i}/{n})", end="\r")

    print(f"\n[Chronos] 예측 완료 | 유효 예측: {np.sum(~np.isnan(predictions))}개")
    return predictions, pred_upper, pred_lower


def generate_signals(prices, predictions):
    """
    예측값을 트레이딩 시그널로 변환합니다.

    시그널: 예측 가격 > 현재 가격 → 1 (매수), 아니면 0 (매도)
    신뢰도: (예측가격 - 현재가격) / 현재가격 (예측 수익률)

    Returns:
        signals: 0/1 배열
        confidence: 예측 수익률 배열
    """
    valid = ~np.isnan(predictions)
    signals = np.zeros(len(prices), dtype=int)
    confidence = np.zeros(len(prices))

    signals[valid] = (predictions[valid] > prices[valid]).astype(int)
    confidence[valid] = (predictions[valid] - prices[valid]) / prices[valid]

    return signals, confidence
