"""通用训练/增量训练/预测输入分派。"""

from __future__ import annotations

import json
from pathlib import Path

from .Execute_pipe import main
from .feature_schema import extract_generic_payload, is_generic_payload


def _target_names(targets: dict) -> list:
    names = []
    for values in (targets or {}).values():
        if isinstance(values, dict):
            for name in values:
                if name not in names:
                    names.append(name)
    return names


def _trained_properties() -> list:
    output_path = Path("./results/property_prediction/training/output.json")
    try:
        with open(output_path, "r", encoding="utf-8") as file:
            names = json.load(file).get("properties_total") or []
        if names:
            return list(names)
    except (OSError, json.JSONDecodeError):
        pass
    model_dir = Path("./artifacts/property_prediction/best_models")
    if not model_dir.exists():
        return []
    return sorted(
        path.name[len("best_model_info_"):-len(".json")]
        for path in model_dir.glob("best_model_info_*.json")
    )


def _save_internal_data(data: dict, output_path: Path, training: bool) -> dict:
    features, targets = extract_generic_payload(data)
    if not isinstance(features, dict) or not features:
        raise ValueError("输入必须包含非空 features（特征）")
    internal = {"generic_features": features}
    if training:
        if not isinstance(targets, dict) or not targets:
            raise ValueError("训练输入必须包含非空 targets（目标）")
        internal["generic_targets"] = targets
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(internal, file, ensure_ascii=False, indent=2)

    config = dict(
        (data.get("train_config") if training else data.get("predict_config"))
        or {}
    )
    if training:
        available = _target_names(targets)
        selected = config.get("targets")
        config["properties"] = list(selected) if selected else available
        unknown = [name for name in config["properties"] if name not in available]
        if unknown:
            raise ValueError(f"训练配置中存在数据未提供的目标: {unknown}")
    else:
        trained = _trained_properties()
        selected = config.get("targets")
        config["properties"] = list(selected) if selected else trained
        unknown = [name for name in config["properties"] if name not in trained]
        if unknown:
            raise ValueError(f"预测配置中存在未训练的目标: {unknown}")
        if not config["properties"]:
            raise RuntimeError("未找到已训练目标，请先执行训练")
    return config


def _common_train_args(config: dict) -> dict:
    return {
        "properties": config.get("properties") or None,
        "property_indices": config.get("property_indices"),
        "test_size": config.get("test_size", 0.2),
        "n_iterations": config.get("n_iterations", 3),
        "n_folds": config.get("n_folds", 4),
        "random_seed": config.get("random_seed", 100),
        "judge": config.get("judge", 0),
        "max_no_improve_rounds": config.get("max_no_improve_rounds", 0),
        "search_method": config.get("search_method", "random"),
        "n_jobs": config.get("n_jobs", 16),
        "debug_dump_csv": config.get("debug_dump_csv", False),
        "outlier_n": config.get("outlier_n"),
        "remove_outliers": config.get("remove_outliers", False),
        "outlier_std_threshold": config.get("outlier_std_threshold", 2.0),
        "outlier_model": config.get("outlier_model", "ridge"),
        "enable_shap": config.get("enable_shap", True),
    }


def run(
    DATA_DIR: Path = Path("./runtime/property_prediction"),
    input_file: str = "./inputs.json",
):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not is_generic_payload(data):
        raise ValueError("仅支持通用 features/targets 输入格式")

    features, targets = extract_generic_payload(data)
    config_block = data.get("train_config") or data.get("predict_config") or {}
    top_level_run_type = data.get("run_type")
    config_run_type = (
        config_block.get("run_type") if isinstance(config_block, dict) else None
    )
    if (top_level_run_type is not None and config_run_type is not None
            and top_level_run_type != config_run_type):
        raise ValueError(
            "顶层 run_type 与 train_config/predict_config.run_type 不一致: "
            f"{top_level_run_type!r} != {config_run_type!r}"
        )
    run_type = top_level_run_type or config_run_type
    if run_type is None:
        run_type = "train_full" if data.get("train_config") is not None or targets is not None else "predict"
    if run_type not in {"train_full", "train_inc", "predict"}:
        raise ValueError(f"run_type 仅支持 train_full/train_inc/predict，当前为: {run_type}")

    if run_type in {"train_full", "train_inc"}:
        config = _save_internal_data(data, DATA_DIR / "train.json", training=True)
        args = _common_train_args(config)
        if run_type == "train_full":
            args.update({
                "model_type": config.get("model_type", "custom"),
                "nn_backend": config.get("nn_backend", "sklearn"),
                "model_list_custom": config.get("model_list_custom", ["xgb", "ridge", "fnn"]),
                "reduction_type": config.get("reduction_type", "custom"),
                "mode_list": config.get(
                    "reduction_list_custom",
                    config.get("mode_list_custom", config.get("mode_list", ["none"])),
                ),
            })
        args["pipeline_type"] = run_type
        main(**args)
        return

    config = _save_internal_data(data, DATA_DIR / "predict.json", training=False)
    main(properties=config["properties"], pipeline_type="predict")


if __name__ == "__main__":
    run()
