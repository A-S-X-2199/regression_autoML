"""通用 JSON 特征/目标数据处理。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .feature_schema import ID_COLUMN, build_feature_frame, build_target_frame


class JSONDataProcessor:
    """将通用 JSON 输入转换为训练和预测所需的 DataFrame。"""

    def __init__(self):
        self._properties_list: Optional[List[str]] = None

    def load_json_data(self, json_path: str | Path) -> Optional[Dict[str, Any]]:
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            print(f"成功加载JSON数据: {json_path}")
            return data
        except Exception as exc:
            print(f"加载JSON数据失败: {exc}")
            return None

    def get_properties_list(
        self,
        json_path: Optional[str | Path] = None,
        training_data: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> List[str]:
        if self._properties_list is not None and not force_refresh:
            return self._properties_list
        if training_data is None and json_path is not None:
            training_data = self.load_json_data(json_path)
        targets = (training_data or {}).get("generic_targets") or {}
        names: List[str] = []
        for values in targets.values():
            if isinstance(values, dict):
                for name in values:
                    if name not in names:
                        names.append(str(name))
        self._properties_list = names
        print(f"从训练数据中提取到 {len(names)} 个目标")
        return names

    @staticmethod
    def _export_correlations(
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        property_list: List[str],
    ) -> None:
        """导出每个目标与特征的 Pearson 相关系数，供图表数据合并。"""
        merged = features_df.merge(targets_df, on=ID_COLUMN, how="inner")
        feature_cols = list(features_df.columns[1:])
        result: Dict[str, Dict[str, float]] = {}
        for target in property_list:
            if target not in merged.columns:
                continue
            correlations = {}
            for feature in feature_cols:
                valid = merged[[feature, target]].dropna()
                if len(valid) < 2 or valid[feature].nunique() < 2 or valid[target].nunique() < 2:
                    value = 0.0
                else:
                    value = float(valid[feature].corr(valid[target]))
                    if not np.isfinite(value):
                        value = 0.0
                correlations[feature] = round(value, 6)
            result[target] = dict(
                sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)[:20]
            )
        if result:
            from .path_config import JSON_RESULT_TRAIN_ROOT
            output = JSON_RESULT_TRAIN_ROOT / "corr_data.json"
            os.makedirs(output.parent, exist_ok=True)
            with open(output, "w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2)

    def process_training_data(
        self,
        property_list: List[str],
        training_json_path: str | Path,
        corr_fig_output_dir: str | Path,
        **_ignored,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = self.load_json_data(training_json_path)
        if not data or "generic_features" not in data:
            raise ValueError("训练数据必须包含 generic_features")
        features_df = build_feature_frame(data["generic_features"])
        targets_df = build_target_frame(
            data.get("generic_targets") or {}, features_df[ID_COLUMN].tolist()
        )
        self.get_properties_list(training_data=data, force_refresh=True)
        self._export_correlations(features_df, targets_df, property_list)
        print(f"通用训练数据处理完成: Features={features_df.shape}, Targets={targets_df.shape}")
        return features_df, targets_df

    def process_prediction_data(
        self,
        prediction_json_path: str | Path,
        **_ignored,
    ) -> pd.DataFrame:
        data = self.load_json_data(prediction_json_path)
        if not data or "generic_features" not in data:
            raise ValueError("预测数据必须包含 generic_features")
        features_df = build_feature_frame(data["generic_features"])
        print(f"通用预测数据处理完成: Features={features_df.shape}")
        return features_df
