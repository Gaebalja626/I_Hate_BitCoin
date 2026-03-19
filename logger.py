"""
로깅 모듈
- 콘솔 출력 + 로그 파일 동시 기록
- 실험 폴더 내 experiment.log 에 저장
- 전역 로그는 results/all_experiments.log 에 누적
"""
import os
import sys
import logging
from datetime import datetime

from config import RESULTS_DIR


def setup_logger(exp_dir=None, name="btc_experiment"):
    """
    로거를 설정합니다.
    - 콘솔 (StreamHandler)
    - 전역 로그 파일 (results/all_experiments.log, 누적)
    - 실험별 로그 파일 (exp_dir/experiment.log, 실험마다 새로)

    Returns:
        logger: logging.Logger 인스턴스
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1) 콘솔 핸들러
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 2) 전역 누적 로그
    os.makedirs(RESULTS_DIR, exist_ok=True)
    global_log = os.path.join(RESULTS_DIR, "all_experiments.log")
    global_fh = logging.FileHandler(global_log, mode="a", encoding="utf-8")
    global_fh.setLevel(logging.INFO)
    global_fh.setFormatter(formatter)
    logger.addHandler(global_fh)

    # 3) 실험별 로그 (exp_dir 지정 시)
    if exp_dir:
        os.makedirs(exp_dir, exist_ok=True)
        exp_log = os.path.join(exp_dir, "experiment.log")
        exp_fh = logging.FileHandler(exp_log, mode="w", encoding="utf-8")
        exp_fh.setLevel(logging.DEBUG)
        exp_fh.setFormatter(formatter)
        logger.addHandler(exp_fh)

    return logger


def add_experiment_log(logger, exp_dir):
    """실험 폴더가 나중에 결정될 때 핸들러를 추가합니다."""
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    os.makedirs(exp_dir, exist_ok=True)
    exp_log = os.path.join(exp_dir, "experiment.log")
    exp_fh = logging.FileHandler(exp_log, mode="w", encoding="utf-8")
    exp_fh.setLevel(logging.DEBUG)
    exp_fh.setFormatter(formatter)
    logger.addHandler(exp_fh)
