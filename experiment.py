"""
실험 관리 모듈
- 실험 폴더 생성, 설정/결과 저장, 실험 목록 조회
"""
import os
import json
from datetime import datetime

from config import RESULTS_DIR


def create_experiment(name="lgbm", config_dict=None):
    """
    실험 폴더를 생성하고 설정을 저장합니다.

    Returns:
        exp_dir: 실험 폴더 경로
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{name}_{timestamp}"
    exp_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    if config_dict:
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    print(f"[실험 생성] {exp_dir}")
    return exp_dir


def save_metrics(exp_dir, result):
    """백테스트 결과를 JSON으로 저장합니다."""
    metrics = {k: v for k, v in result.items() if k != "predictions"}
    # portfolio_values는 리스트로 유지
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  [저장] {metrics_path}")


def list_experiments():
    """기존 실험 목록을 출력합니다."""
    if not os.path.exists(RESULTS_DIR):
        print("실험 결과 없음")
        return []

    experiments = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        exp_dir = os.path.join(RESULTS_DIR, d)
        if not os.path.isdir(exp_dir):
            continue
        metrics_path = os.path.join(exp_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                m = json.load(f)
            experiments.append({
                "name": d,
                "path": exp_dir,
                "return": m.get("model_return_pct"),
                "buyhold": m.get("buyhold_return_pct"),
                "sharpe": m.get("sharpe_ratio"),
                "mdd": m.get("mdd_pct"),
                "accuracy": m.get("accuracy"),
                "trades": m.get("total_trades"),
            })

    if experiments:
        print(f"\n{'='*80}")
        print(f"{'실험명':<35} {'수익률':>8} {'B&H':>8} {'Sharpe':>7} {'MDD':>7} {'정확도':>7} {'거래':>5}")
        print(f"{'-'*80}")
        for e in experiments:
            print(
                f"{e['name']:<35} "
                f"{e['return']:>+7.2f}% "
                f"{e['buyhold']:>+7.2f}% "
                f"{e['sharpe']:>7.2f} "
                f"{e['mdd']:>6.2f}% "
                f"{e['accuracy']:>7.4f} "
                f"{e['trades']:>5d}"
            )
        print(f"{'='*80}")
    else:
        print("실험 결과 없음")

    return experiments
