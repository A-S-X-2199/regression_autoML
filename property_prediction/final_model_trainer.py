import json
import os
import pickle
import traceback
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .pca_module import PCAReducer
from .pls_module import PLSReducer
from .svd_module import SVDReducer
from .model_optimizer_main import ModelOptimizer
from .pytorch_module import PyTorchModelSaver
from .model_input import prepare_model_input


def _safe_prop(prop: str) -> str:
    return prop.replace("（", "_").replace("）", "_").replace("/", "_")


# 复用 ModelOptimizer 的模型工厂（仅用 create_model_with_params，不触发训练）
_optimizer = ModelOptimizer()

_NN_TYPES = ['fnn', 'deep_fnn', 'simple_fnn', 'resnet']


def _build_and_fit_model(model_type: str, hyperparams: Dict, X_model: pd.DataFrame,
                         y_final: pd.Series, nn_backend: str) -> object:
    """基于保存的超参数重建模型并 fit（复刻 train_final_model 的参数处理逻辑）"""
    _optimizer.nn_backend = nn_backend
    if model_type not in _optimizer.model_configs:
        _optimizer.model_configs[model_type] = {}

    final_params = dict(hyperparams or {})
    final_params.pop('reduction_ratio', None)

    n_samples = len(y_final)
    # sklearn 神经网络 batch_size 计算
    if model_type in _NN_TYPES and 'batch_size_ratio' in final_params:
        batch_size_ratio = final_params.pop('batch_size_ratio')
        if final_params.get('early_stopping', False):
            val_frac = final_params.get('validation_fraction', 0.1)
            effective = int(n_samples * (1 - val_frac))
        else:
            effective = n_samples
        batch_size = max(1, int(effective * batch_size_ratio))
        batch_size = min(batch_size, effective)
        if effective >= 2:
            batch_size = max(2, batch_size)
        final_params['batch_size'] = batch_size

    # KNN 安全截断
    if model_type == 'knn' and 'n_neighbors' in final_params:
        if final_params['n_neighbors'] > n_samples:
            final_params['n_neighbors'] = max(1, n_samples)

    if model_type in _NN_TYPES and nn_backend == 'pytorch':
        input_size = X_model.shape[1]
        model = _optimizer.create_model_with_params(model_type, final_params, input_size)
        batch_size = final_params.get('batch_size')
        if batch_size is None:
            batch_size = max(2, int(n_samples * final_params.get('batch_size_ratio', 0.1)))
        model.fit(
            X_model, y_final,
            epochs=final_params.get('epochs', 1000),
            batch_size=batch_size,
            early_stopping=final_params.get('early_stopping', True),
            patience=final_params.get('patience', 50),
            verbose=True,
        )
        return model

    model = _optimizer.create_model_with_params(model_type, final_params)
    model.fit(prepare_model_input(model_type, X_model), y_final)
    return model


def train_final_models(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    best_models_dir: str,
    final_models_dir: str,
    properties: Optional[List[str]] = None,
) -> List[str]:
    """基于全量数据（训练集+测试集全部样本）重新训练各性能项点的最终模型。

    背景：训练阶段使用 train/test 分割评估模型，最终部署模型应在全部样本上
    重新拟合，以利用所有可用数据。本函数根据 best_models 中保存的模型信息
    （模型类型/超参数/特征/降维方式），对每个项点：
      1. 取该性质全部有效样本（目标值非 NaN）
      2. 按通用前/后部分特征结构处理特征
      3. 标准化：对全量样本拟合 StandardScaler
      4. 降维（若采用）：对全量样本拟合降维器
      5. 重建模型并在全量数据上训练
    结果保存到 final_models_dir：
      - standardization_params_{prop}.pkl   标准化参数
      - {reduction_type}_params_{prop}.pkl  降维参数（若采用降维）
      - best_model_{prop}.pkl / .pth        最终模型
    info（best_model_info / best_features / best_params）仍从 best_models_dir 读取，
    预测链路中 info 仍读原位置，模型与标准化/降维参数优先从 final_models_dir 读取。

    异常点剔除由调用方在数据准备阶段对全量样本完成（与训练一致），本函数直接使用。
    """
    os.makedirs(final_models_dir, exist_ok=True)

    if properties is None:
        properties = []
        if os.path.isdir(best_models_dir):
            for f in os.listdir(best_models_dir):
                if f.startswith('best_model_info_') and f.endswith('.json'):
                    properties.append(f.replace('best_model_info_', '').replace('.json', ''))

    trained = []
    for prop in properties:
        safe = _safe_prop(prop)
        info_path = os.path.join(best_models_dir, f'best_model_info_{safe}.json')
        features_path = os.path.join(best_models_dir, f'best_features_{safe}.json')
        if not os.path.exists(info_path) or not os.path.exists(features_path):
            print(f"  警告: 项点 '{prop}' 缺少 best_model_info/best_features，跳过最终模型训练")
            continue
        if prop not in targets_df.columns:
            print(f"  警告: 项点 '{prop}' 不在目标数据中，跳过")
            continue

        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        with open(features_path, 'r', encoding='utf-8') as f:
            final_features = json.load(f)

        try:
            print(f"\n=== 最终模型训练（全量数据）: {prop} ===")
            y = targets_df[prop]
            valid_mask = y.notna()
            X_full = features_df[valid_mask].copy().reset_index(drop=True)
            y_full = y[valid_mask].copy().reset_index(drop=True)
            if len(X_full) == 0:
                print(f"  警告: 项点 '{prop}' 无有效样本，跳过")
                continue
            print(f"  全量有效样本数: {len(X_full)}")

            # 特征准备：剔除样本编号列
            X = X_full.iloc[:, 1:].copy()

            # 标准化：对全量样本 fit
            X_values = X.values.astype(np.float64)
            scaler = StandardScaler()
            X_scaled_np = scaler.fit_transform(X_values)
            X_scaled = pd.DataFrame(X_scaled_np, columns=X.columns)
            std_params = {
                'mean': scaler.mean_,
                'std': scaler.scale_,
                'feature_names': X.columns.tolist(),
                'fitted_on_train_size': len(X_full),
                'property_name': prop,
            }
            std_path = os.path.join(final_models_dir, f'standardization_params_{safe}.pkl')
            with open(std_path, 'wb') as f:
                pickle.dump(std_params, f)
            print(f"  标准化参数已保存（基于全量 {len(X_full)} 样本）: {std_path}")

            # 降维：对全量样本 fit（参数保存至 final_models_dir，格式与训练一致）
            reduction_type = model_info.get('reduction_type', 'none')
            reduction_ratio = model_info.get('reduction_ratio', 0.5)
            reduction_features = model_info.get('reduction_features', 20)
            if reduction_type and reduction_type != 'none':
                if reduction_type == 'svd':
                    reducer = SVDReducer(reduction_features, svd_type='svd')
                elif reduction_type == 'tsvd':
                    reducer = SVDReducer(reduction_features, svd_type='tsvd')
                elif reduction_type == 'pca':
                    reducer = PCAReducer(reduction_features)
                elif reduction_type == 'pls':
                    reducer = PLSReducer(reduction_features)
                else:
                    reducer = None
                if reducer is not None:
                    X_final, _ = reducer.reduce_full_data(
                        final_models_dir, final_models_dir, X_scaled, y_full,
                        ratio=reduction_ratio, property_name=prop)
                    print(f"  应用{reduction_type.upper()}降维（ratio={reduction_ratio}），"
                          f"降维后特征数: {X_final.shape[1]}")
                else:
                    X_final = X_scaled
            else:
                X_final = X_scaled

            X_final = X_final.reset_index(drop=True)
            y_model = y_full.reset_index(drop=True)

            # 按 best_features 选取最终特征列（保证与训练时模型输入一致）
            available = [c for c in final_features if c in X_final.columns]
            missing = [c for c in final_features if c not in X_final.columns]
            if missing:
                print(f"  警告: 特征列表缺失 {len(missing)} 列（{missing[:5]}...），使用可用特征")
            X_model = X_final[available] if available else X_final
            print(f"  模型训练特征数: {X_model.shape[1]}")

            # 重建并训练最终模型
            model_type = model_info.get('best_model')
            nn_backend = model_info.get('nn_backend', 'sklearn')
            model = _build_and_fit_model(model_type, model_info.get('hyperparameters'),
                                         X_model, y_model, nn_backend)

            if model_type in _NN_TYPES and nn_backend == 'pytorch':
                model_path = os.path.join(final_models_dir, f'best_model_{safe}.pth')
                model_config = {
                    'model_type': model_type,
                    'input_size': X_model.shape[1],
                    'hidden_sizes': (model_info.get('hyperparameters') or {}).get(
                        'hidden_layer_sizes', (50,)),
                    'activation': (model_info.get('hyperparameters') or {}).get(
                        'activation', 'relu'),
                    'dropout_rate': (model_info.get('hyperparameters') or {}).get(
                        'dropout_rate', 0.0),
                    'batch_norm': (model_info.get('hyperparameters') or {}).get(
                        'batch_norm', False),
                }
                PyTorchModelSaver.save_model(model.model, model_path, model_config)
                print(f"  最终模型已保存: {model_path}")
            else:
                model_path = os.path.join(final_models_dir, f'best_model_{safe}.pkl')
                joblib.dump(model, model_path)
                print(f"  最终模型已保存: {model_path}")

            trained.append(prop)
        except Exception as e:
            print(f"  错误: 项点 '{prop}' 最终模型训练失败: {e}")
            print(traceback.format_exc())

    print(f"\n最终模型训练完成: {len(trained)}/{len(properties)} 个项点")
    return trained
