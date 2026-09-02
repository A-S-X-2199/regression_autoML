"""模型适配层：保持外部特征名，同时规避个别模型的列名限制。"""

from __future__ import annotations

import pandas as pd


def prepare_model_input(model_type: str, data):
    """LightGBM 训练时使用无列名矩阵，避免其 JSON 特征名字符限制。"""
    if model_type == "lgbm" and isinstance(data, pd.DataFrame):
        return data.to_numpy()
    return data
