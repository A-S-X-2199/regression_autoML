"""通用多目标回归预测器命令行入口。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from threading import Thread

import psutil


class TeeOutput:
    """同时写终端和 UTF-8 日志；warning 仅显示在终端。"""

    def __init__(self, log_path, original):
        self.log = open(log_path, "w", encoding="utf-8")
        self.original = original
        self._buffer = ""

    def write(self, value):
        self.original.write(value)
        self._buffer += value
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not line.lstrip().startswith("WARNING:"):
                self.log.write(line + "\n")

    def flush(self):
        self.original.flush()
        self.log.flush()

    def close(self):
        if self._buffer and not self._buffer.lstrip().startswith("WARNING:"):
            self.log.write(self._buffer)
        self.log.close()


def _is_prediction(data: dict) -> bool:
    config = data.get("train_config") or data.get("predict_config") or {}
    top_level_run_type = data.get("run_type")
    config_run_type = config.get("run_type") if isinstance(config, dict) else None
    if (top_level_run_type is not None and config_run_type is not None
            and top_level_run_type != config_run_type):
        raise ValueError(
            "顶层 run_type 与 train_config/predict_config.run_type 不一致: "
            f"{top_level_run_type!r} != {config_run_type!r}"
        )
    run_type = top_level_run_type or config_run_type
    if run_type:
        return run_type == "predict"
    payload = data.get("dataset", data)
    has_features = isinstance(payload, dict) and ("features" in payload or "特征" in payload)
    has_targets = isinstance(payload, dict) and ("targets" in payload or "目标" in payload)
    if has_features:
        return not has_targets and "train_config" not in data
    return "predict" in data


def _clean_nan(value):
    if isinstance(value, dict):
        return {key: _clean_nan(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_nan(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def model_run(inputs_file: str = "./inputs.json") -> None:
    """执行训练或预测，并把资源统计合并进结果 JSON。"""
    with open(inputs_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    prediction = _is_prediction(data)
    config = data.get("train_config") or data.get("predict_config") or {}
    n_jobs = config.get("n_jobs", config.get("njobs", psutil.cpu_count()))
    target_json = Path(
        "./results/property_prediction/prediction/output.json"
        if prediction else "./results/property_prediction/training/output.json"
    )

    process = psutil.Process()
    samples = {"mem": [], "cpu": []}
    running = True

    def monitor():
        while running:
            samples["mem"].append(process.memory_info().rss / 1024 / 1024)
            samples["cpu"].append(process.cpu_percent(interval=0.1))
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        data_dir = Path("./runtime/property_prediction")
        data_dir.mkdir(parents=True, exist_ok=True)
        if prediction and target_json.exists():
            target_json.unlink()
        from property_prediction.json_input_to_ml import run
        run(data_dir, inputs_file)
    finally:
        running = False
        monitor_thread.join()

    cpu = samples["cpu"] or [0.0]
    mem = samples["mem"] or [0.0]
    resource = {
        "resource": {
            "cpu": {"n_jobs": n_jobs, "avg": f"{sum(cpu) / len(cpu):.2f}%", "max": f"{max(cpu):.2f}%"},
            "mem": {"avg": f"{sum(mem) / len(mem):.2f}MB", "max": f"{max(mem):.2f}MB"},
        }
    }
    target_json.parent.mkdir(parents=True, exist_ok=True)
    if target_json.exists():
        with open(target_json, "r", encoding="utf-8") as file:
            output = json.load(file)
        output.update(resource)
    else:
        output = resource
    with open(target_json, "w", encoding="utf-8") as file:
        json.dump(_clean_nan(output), file, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> None:
    original_stderr = sys.stderr
    log_path = Path("./results/property_prediction/latest_run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee = TeeOutput(log_path, sys.stdout)
    sys.stdout = tee
    sys.stderr = tee

    import warnings

    def showwarning(message, category, filename, lineno, file=None, line=None):
        original_stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
        original_stderr.flush()

    warnings.showwarning = showwarning
    parser = argparse.ArgumentParser(description="训练或运行通用多目标回归预测器")
    parser.add_argument("--inputs_file", default="./inputs.json", help="训练或预测 JSON 输入文件")
    args = parser.parse_args()
    try:
        model_run(args.inputs_file)
    finally:
        sys.stdout = tee.original
        sys.stderr = original_stderr
        tee.close()


if __name__ == "__main__":
    main()
