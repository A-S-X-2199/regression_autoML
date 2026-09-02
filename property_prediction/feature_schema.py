"""通用预测器的输入结构与特征分组工具。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, Mapping, Tuple

import pandas as pd


ID_COLUMN = "样本编号"
FRONT_PREFIX = "FRONT::"
BACK_PREFIX = "BACK::"


def _section(record: Mapping[str, Any], chinese: str, english: str) -> Mapping[str, Any]:
    value = record.get(chinese, record.get(english, {}))
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"特征的‘{chinese}’必须是对象")
    return value


def extract_generic_payload(data: Mapping[str, Any]) -> Tuple[dict | None, dict | None]:
    """从顶层或 dataset 中提取 features/targets（同时接受中英文键）。"""
    payload = data.get("dataset", data)
    if not isinstance(payload, Mapping):
        return None, None
    features = payload.get("features", payload.get("特征"))
    targets = payload.get("targets", payload.get("目标"))
    return features, targets


def is_generic_payload(data: Mapping[str, Any]) -> bool:
    features, _ = extract_generic_payload(data)
    return isinstance(features, Mapping)


def build_feature_frame(features: Mapping[str, Any]) -> pd.DataFrame:
    """将样本字典转为宽表；前后部分均可缺省，稀疏字段补 0。"""
    if not isinstance(features, Mapping) or not features:
        raise ValueError("features（特征）必须是非空的‘样本编号 -> 特征对象’映射")

    rows = []
    ordered_columns: "OrderedDict[str, None]" = OrderedDict()
    for sample_id, raw_record in features.items():
        if raw_record is None:
            raw_record = {}
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"样本 '{sample_id}' 的特征必须是对象")
        front = _section(raw_record, "前部分", "front")
        back = _section(raw_record, "后部分", "back")
        if not front and not back:
            raise ValueError(f"样本 '{sample_id}' 的前部分和后部分不能同时为空")

        row: Dict[str, Any] = {ID_COLUMN: str(sample_id)}
        for name, value in front.items():
            col = f"{FRONT_PREFIX}{name}"
            ordered_columns.setdefault(col, None)
            row[col] = value
        for name, value in back.items():
            col = f"{BACK_PREFIX}{name}"
            ordered_columns.setdefault(col, None)
            row[col] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    columns = [ID_COLUMN, *ordered_columns.keys()]
    frame = frame.reindex(columns=columns)
    for col in columns[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame


def build_target_frame(targets: Mapping[str, Any], sample_ids: Iterable[str]) -> pd.DataFrame:
    """按特征样本顺序构建目标宽表；目标缺失保留为 NaN。"""
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("训练输入必须提供非空 targets（目标）")
    target_names: "OrderedDict[str, None]" = OrderedDict()
    for sample_id, values in targets.items():
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise TypeError(f"样本 '{sample_id}' 的目标必须是对象")
        for name in values:
            target_names.setdefault(str(name), None)
    if not target_names:
        raise ValueError("targets（目标）中没有目标字段")

    rows = []
    for sample_id in sample_ids:
        values = targets.get(sample_id, targets.get(str(sample_id), {})) or {}
        row = {ID_COLUMN: str(sample_id)}
        for name in target_names:
            row[name] = values.get(name)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for col in target_names:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def split_feature_columns(columns: Iterable[Any], id_column: str = ID_COLUMN):
    """按输入中的显式 FRONT/BACK 标记拆分前、后部分特征。"""
    cols = [c for c in columns if c != id_column]
    front = [c for c in cols if str(c).startswith(FRONT_PREFIX)]
    back = [c for c in cols if str(c).startswith(BACK_PREFIX)]
    # 模型变换后产生的无前缀列不再参与原始分组，按前部分保留。
    front.extend(c for c in cols if c not in front and c not in back)
    return front, back


def display_feature_name(name: Any) -> str:
    value = str(name)
    for prefix in (FRONT_PREFIX, BACK_PREFIX):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value
