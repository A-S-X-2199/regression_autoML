"""按目标独立划分并标准化通用特征数据。"""

from __future__ import annotations

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data_per_property(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    property_name: str,
    standardization_params_dir: str = "reduction_params",
    test_size: float = 0.2,
    random_state: int = 42,
    standardize: bool = True,
) -> tuple:
    """为一个目标过滤缺失值，并独立完成训练/测试划分与标准化。"""
    if property_name not in df_targets.columns:
        raise ValueError(f"目标 '{property_name}' 不在目标数据中")
    if len(df_features) != len(df_targets):
        raise ValueError(
            f"特征与目标样本数不一致: {len(df_features)} != {len(df_targets)}"
        )
    if not 0 <= test_size < 1:
        raise ValueError("test_size 必须在 [0, 1) 范围内")

    id_col = df_features.columns[0]
    valid_mask = df_targets[property_name].notna()
    X_valid = df_features.loc[valid_mask].copy().reset_index(drop=True)
    y_valid = df_targets.loc[valid_mask, property_name].copy().reset_index(drop=True)

    print(f"\n--- 按目标划分: {property_name} ---")
    print(f"原始样本数: {len(df_features)}, 有效样本数: {len(X_valid)}")
    if X_valid.empty:
        raise ValueError(f"目标 '{property_name}' 的值全部为空，无法训练")

    indices = list(range(len(X_valid)))
    if test_size == 0:
        train_indices, test_indices = indices, []
    else:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

    X_train = X_valid.iloc[train_indices].reset_index(drop=True)
    y_train = y_valid.iloc[train_indices].reset_index(drop=True)
    X_test = X_valid.iloc[test_indices].reset_index(drop=True)
    y_test = y_valid.iloc[test_indices].reset_index(drop=True)
    print(f"训练集: {len(X_train)} 样本 | 测试集: {len(X_test)} 样本")

    if not standardize:
        return X_train, y_train, X_test, y_test

    feature_names = X_train.columns[1:].tolist()
    if not feature_names:
        raise ValueError("输入数据没有可用于建模的特征")

    scaler = StandardScaler()
    train_values = scaler.fit_transform(X_train[feature_names])
    X_train_scaled = pd.DataFrame(train_values, columns=feature_names)
    X_train_scaled.insert(0, id_col, X_train[id_col].values)

    if X_test.empty:
        X_test_scaled = X_test.copy()
    else:
        test_values = scaler.transform(X_test[feature_names])
        X_test_scaled = pd.DataFrame(test_values, columns=feature_names)
        X_test_scaled.insert(0, id_col, X_test[id_col].values)

    safe_name = property_name.replace("（", "_").replace("）", "_").replace("/", "_")
    os.makedirs(standardization_params_dir, exist_ok=True)
    params_path = os.path.join(
        standardization_params_dir,
        f"standardization_params_{safe_name}.pkl",
    )
    with open(params_path, "wb") as file:
        pickle.dump(
            {
                "mean": scaler.mean_,
                "std": scaler.scale_,
                "feature_names": feature_names,
                "fitted_on_train_size": len(X_train),
                "property_name": property_name,
            },
            file,
        )
    print(f"目标标准化参数已保存: {params_path}")
    return X_train_scaled, y_train, X_test_scaled, y_test
