"""
SHAP 可解释性分析模块
功能：训练完成后对每个性质生成 SHAP 分析图（全局摘要图、局部瀑布图、特征依赖图）
直接分析每个目标最佳模型实际使用的全部特征。
输出：PNG 图片到 results/property_prediction/visualizations/SHAP分析/，数值数据先导出到 shap_data.json，流程末尾统一合并为 chart_data.json。
"""

import os
import json
import base64
import pickle
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.pipeline import Pipeline
from .feature_schema import split_feature_columns


# 修复 shap 与 sklearn 1.9+ 兼容性：Pipeline.feature_names_in_ 是只读 property，
# 但 shap.TreeExplainer 内部会尝试赋值。这里 patch 成可写。
_pipeline_prop = Pipeline.__dict__.get('feature_names_in_')
if _pipeline_prop is not None:
    try:
        Pipeline.feature_names_in_ = property(
            _pipeline_prop.fget,
            _pipeline_prop.fset if _pipeline_prop.fset else lambda self, v: setattr(self, '_feature_names_in_', v),
            _pipeline_prop.fdel,
        )
    except Exception:
        pass

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ["SimHei", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False

TREE_MODELS = {'xgb', 'rf', 'lgbm', 'gbr', 'catboost', 'extra_trees', 'hist_gbdt', 'adaboost', 'gbdt'}
LINEAR_MODELS = {'ridge', 'lasso', 'elasticnet', 'bayesian_ridge', 'linear'}
NN_MODELS = {'fnn', 'deep_fnn', 'simple_fnn', 'resnet'}


def _cleanup_catboost_info():
    if os.path.exists("catboost_info"):
        try:
            shutil.rmtree("catboost_info")
        except Exception:
            pass


def _safe_prop_name(prop_name: str) -> str:
    safe = prop_name
    for ch in ["（", "）", "/", "\\"]:
        safe = safe.replace(ch, "_")
    return safe


def _load_model_and_features(model_dir: str, prop_name: str) -> Tuple[object, List[str], str, str]:
    safe_prop = _safe_prop_name(prop_name)
    info_path = os.path.join(model_dir, f"best_model_info_{safe_prop}.json")
    features_path = os.path.join(model_dir, f"best_features_{safe_prop}.json")

    if not os.path.exists(info_path):
        raise FileNotFoundError(f"模型信息文件不存在: {info_path}")

    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    nn_backend = info.get('nn_backend', 'sklearn')
    model_type = info.get('best_model', 'unknown')
    if os.path.exists(features_path):
        with open(features_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        if not isinstance(features, list):
            raise TypeError(f"模型特征文件必须是列表: {features_path}")
        print(f"  [SHAP DEBUG] {prop_name}: loaded {len(features)} features from {features_path}")
    else:
        features = info.get('features', [])
        print(f"  [SHAP DEBUG] {prop_name}: best_features 文件不存在，回退模型信息中的 {len(features)} 个特征")
    reduction_type = info.get('reduction_type', 'none')

    pth_path = os.path.join(model_dir, f"best_model_{safe_prop}.pth")
    pkl_path = os.path.join(model_dir, f"best_model_{safe_prop}.pkl")

    if nn_backend == 'pytorch' and model_type in NN_MODELS and os.path.exists(pth_path):
        from .pytorch_module import PyTorchModelSaver
        model = PyTorchModelSaver.load_model(pth_path, device='cpu')
    elif os.path.exists(pkl_path):
        import joblib
        model = joblib.load(pkl_path)
    elif os.path.exists(pth_path):
        from .pytorch_module import PyTorchModelSaver
        model = PyTorchModelSaver.load_model(pth_path, device='cpu')
    else:
        raise FileNotFoundError(
            f"模型文件不存在: {pkl_path} 或 {pth_path}"
        )

    return model, features, model_type, reduction_type


def _reconstruct_training_features(
    train_features_df: pd.DataFrame,
    prop_name: str,
    training_params_dir: str,
    reduction_type: str = 'none',
) -> Tuple[pd.DataFrame, List[str]]:
    """
    从原始训练特征重建模型实际使用的特征（前部分 + 降维成分）
    支持 PCA / PLS / SVD / TSVD
    返回: (完整特征DataFrame, 前部分特征列名列表)
    """
    safe_prop = _safe_prop_name(prop_name)

    df = train_features_df.copy()
    feature_cols = list(df.columns[1:])
    front_cols, _back_cols = split_feature_columns(df.columns)

    # 加载标准化参数（训练时 split_data_per_property 做了标准化）
    std_params_path = os.path.join(training_params_dir, f"standardization_params_{safe_prop}.pkl")
    scaler = None
    if os.path.exists(std_params_path):
        try:
            with open(std_params_path, 'rb') as f:
                std_params = pickle.load(f)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = std_params['mean']
            scaler.scale_ = std_params['std']
            scaler.feature_names_in_ = np.array(std_params['feature_names'])
            print(f"  [SHAP DEBUG] {prop_name}: loaded scaler from {std_params_path}")
        except Exception as e:
            print(f"  [SHAP DEBUG] {prop_name}: failed to load scaler: {e}")

    def _standardized_raw_features() -> pd.DataFrame:
        """重建未降维模型输入；也作为无后部分特征时的安全回退。"""
        X_full = df[feature_cols].copy()
        if scaler is not None:
            common_cols = [c for c in feature_cols if c in std_params['feature_names']]
            if common_cols:
                idx_map = {name: i for i, name in enumerate(std_params['feature_names'])}
                indices = [idx_map[c] for c in common_cols]
                raw = X_full[common_cols].values.astype(np.float64)
                scaled = (raw - scaler.mean_[indices]) / scaler.scale_[indices]
                X_full[common_cols] = scaled
        return X_full

    if reduction_type == 'none':
        return _standardized_raw_features(), front_cols

    red_type = reduction_type
    prefix_map = {'pca': 'PCA_主成分', 'pls': 'PLS_主成分', 'svd': 'SVD_主成分', 'tsvd': 'SVD_主成分'}
    prefix = prefix_map.get(red_type, 'PCA_主成分')
    filename = f"{red_type}_params_{safe_prop}.pkl"
    param_path = os.path.join(training_params_dir, filename)

    if not os.path.exists(param_path):
        print(
            f"  [SHAP DEBUG] {prop_name}: 未找到 {red_type} 参数；"
            "按未降维原始特征回退（适用于后部分为空的模型）"
        )
        return _standardized_raw_features(), front_cols

    with open(param_path, 'rb') as f:
        params = pickle.load(f)

    transformer_key = 'svd' if red_type == 'tsvd' else red_type
    transformer = params.get(transformer_key)
    if transformer is None:
        print(f"  [SHAP DEBUG] {prop_name}: 降维器为空，按未降维原始特征回退")
        return _standardized_raw_features(), front_cols

    front_columns = params.get('front_columns', [])
    back_columns = params.get('back_columns', [])

    if not back_columns:
        back_columns = [c for c in feature_cols if c not in front_columns]

    missing_front = [c for c in front_columns if c not in df.columns]
    missing_back = [c for c in back_columns if c not in df.columns]
    if missing_front or missing_back:
        print(f"  [SHAP DEBUG] {prop_name}: {red_type} params found but columns missing")
        if missing_front:
            print(f"    missing_front ({len(missing_front)}): {missing_front[:5]}...")
        if missing_back:
            print(f"    missing_back ({len(missing_back)}): {missing_back[:5]}...")
        return pd.DataFrame(index=df.index), front_cols

    print(f"  [SHAP DEBUG] {prop_name}: using {red_type} params, front={len(front_columns)}, back={len(back_columns)}")
    n_components = params.get('n_components', transformer.n_components if hasattr(transformer, 'n_components') else 0)

    # 应用标准化（训练时数据经过了 StandardScaler）
    front_df_raw = df[front_columns].copy() if front_columns else pd.DataFrame(index=df.index)
    back_df_raw = df[back_columns].copy()

    if scaler is not None:
        # 标准化 front columns
        if front_columns:
            common_front = [c for c in front_columns if c in std_params['feature_names']]
            if common_front:
                idx_map = {name: i for i, name in enumerate(std_params['feature_names'])}
                front_indices = [idx_map[c] for c in common_front]
                raw_f = front_df_raw[common_front].values.astype(np.float64)
                scaled_f = (raw_f - scaler.mean_[front_indices]) / scaler.scale_[front_indices]
                for j, c in enumerate(common_front):
                    front_df_raw[c] = scaled_f[:, j]
        # 标准化 back columns
        common_back = [c for c in back_columns if c in std_params['feature_names']]
        if common_back:
            idx_map = {name: i for i, name in enumerate(std_params['feature_names'])}
            back_indices = [idx_map[c] for c in common_back]
            raw_b = back_df_raw[common_back].values.astype(np.float64)
            scaled_b = (raw_b - scaler.mean_[back_indices]) / scaler.scale_[back_indices]
            for j, c in enumerate(common_back):
                back_df_raw[c] = scaled_b[:, j]

    front_df = front_df_raw.reset_index(drop=True) if front_columns else pd.DataFrame(index=df.index).reset_index(drop=True)
    back_data = back_df_raw.values.astype(np.float64)
    transformed = transformer.transform(back_data)
    comp_columns = [f'{prefix}{i+1}' for i in range(transformed.shape[1])]
    comp_df = pd.DataFrame(transformed, columns=comp_columns)

    X_full = pd.concat([front_df, comp_df], axis=1)
    return X_full, front_columns


def _get_shap_explainer(model, X_sample: pd.DataFrame, model_type: str):
    try:
        import shap
    except ImportError:
        raise ImportError("请安装 shap 库: pip install shap")

    # 如果是 Pipeline，解包取出最终 estimator，避免 shap 设置 feature_names_in_ 报错
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
        print(f"  [SHAP DEBUG] Pipeline detected (isinstance), estimator={type(estimator).__name__}")
    elif hasattr(model, 'steps') and len(model.steps) > 0:
        estimator = model.steps[-1][1]
        print(f"  [SHAP DEBUG] Pipeline-like detected (hasattr steps), estimator={type(estimator).__name__}")
    elif hasattr(model, 'named_steps'):
        vals = list(model.named_steps.values())
        estimator = vals[-1] if vals else model
        print(f"  [SHAP DEBUG] Pipeline-like detected (named_steps), estimator={type(estimator).__name__}")
    else:
        estimator = model
        print(f"  [SHAP DEBUG] Not a Pipeline, model type={type(model).__name__}")

    if model_type in TREE_MODELS:
        try:
            print(f"  [SHAP DEBUG] Trying TreeExplainer with estimator {type(estimator).__name__}")
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_sample)
            print(f"  [SHAP DEBUG] TreeExplainer succeeded")
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            return explainer, shap_values, X_sample
        except Exception as e:
            print(f"  [SHAP DEBUG] TreeExplainer failed: {e}, falling back to KernelExplainer")
            pass  # fall through to KernelExplainer
    elif model_type in LINEAR_MODELS:
        explainer = shap.LinearExplainer(estimator, X_sample)
        shap_values = explainer.shap_values(X_sample)
        return explainer, shap_values, X_sample
    elif model_type in NN_MODELS:
        pass  # use KernelExplainer below
    else:
        try:
            print(f"  [SHAP DEBUG] Unknown model type '{model_type}', trying TreeExplainer")
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            return explainer, shap_values, X_sample
        except Exception:
            pass  # fall through to KernelExplainer

    # 最终兜底：使用 KernelExplainer（使用原始 model.predict 以保留 Pipeline transform）
    print(f"  [SHAP DEBUG] Using KernelExplainer")
    n_total = len(X_sample)
    # KernelExplainer 开销约为 n_eval × nsamples 次模型调用，大样本时耗时爆炸，
    # 因此对 background/eval 样本量设上限（全局 summary 基于最多 EVAL_CAP 个样本）。
    bg_cap, eval_cap = 100, 100
    n_bg = min(n_total, bg_cap)
    bg_indices = np.random.choice(n_total, n_bg, replace=False)
    background = X_sample.iloc[bg_indices].reset_index(drop=True)
    explainer = shap.KernelExplainer(model.predict, background)
    n_eval = min(n_total, eval_cap)
    X_eval = X_sample.iloc[:n_eval].reset_index(drop=True)
    nsamples = max(200, n_eval * 4)
    print(f"  [SHAP DEBUG] KernelExplainer: bg={n_bg}, eval={n_eval}, nsamples={nsamples}")
    shap_values = explainer.shap_values(X_eval, nsamples=nsamples)
    return explainer, shap_values, X_eval


def _fig_to_base64(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64_str


def generate_shap_for_property(
    model_dir: str,
    prop_name: str,
    train_features_df: pd.DataFrame,
    training_params_dir: str,
    output_dir: str,
) -> Dict[str, str]:
    """
    为单个性质生成 SHAP 分析图（严格使用该模型保存的全部特征）

    :param model_dir: 模型保存目录 (best_models/)
    :param prop_name: 性质名称
    :param train_features_df: 原始训练集特征 DataFrame
    :param training_params_dir: 降维参数保存目录
    :param output_dir: 图片输出目录
    :return: {图片类型: base64字符串}
    """
    import shap

    os.makedirs(output_dir, exist_ok=True)
    safe_prop = _safe_prop_name(prop_name)
    result = {}

    try:
        model, features, model_type, reduction_type = _load_model_and_features(model_dir, prop_name)

        X_full, front_cols = _reconstruct_training_features(
            train_features_df, prop_name, training_params_dir, reduction_type
        )

        if X_full.empty:
            print(f"  [SHAP] 性质 '{prop_name}': 模型特征重建失败，跳过")
            return result

        if not front_cols:
            # 允许前部分为空：此时使用后部分原始特征或降维主成分。
            usable_cols = [c for c in features if c in X_full.columns]
            if not usable_cols:
                print(f"  [SHAP] 性质 '{prop_name}': 可用特征为空，跳过")
                return result
            print(f"  [SHAP] 性质 '{prop_name}': 前部分为空，使用模型全部 {len(usable_cols)} 个特征")
            X = X_full[usable_cols].copy()
            model_type_final = model_type
            explainer, shap_values, X_eval = _get_shap_explainer(model, X, model_type_final)
            if hasattr(shap_values, 'values'):
                shap_array = shap_values.values
            else:
                shap_array = np.array(shap_values)
            if shap_array.ndim == 3:
                shap_array = shap_array[:, :, 0]
            X_display = X_eval if X_eval is not None else X
            shap_display = shap_array
            n_display = len(usable_cols)

            # 1. 全局摘要图
            shap.summary_plot(shap_display, X_display, show=False)
            fig = plt.gcf()
            fig.set_size_inches(12, 8)
            fig.tight_layout()
            summary_path = os.path.join(output_dir, f"{safe_prop}_summary.png")
            fig.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
            result['SHAP_summary'] = _fig_to_base64(fig)
            plt.close('all')
            print(f"  [SHAP] {prop_name}: 摘要图已保存 ({n_display} 个特征)")

            # 2. 瀑布图
            sample_idx = np.random.randint(0, len(X_display))
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_val = float(explainer.expected_value[0])
            else:
                base_val = float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0

            # ===== 调试：验证 model.predict 是否被正确调用 =====
            print(f"  [SHAP DEBUG] ========== waterfall 模型调用验证: {prop_name} ==========")
            print(f"  [SHAP DEBUG] 模型类型: {type(model).__name__}")
            all_preds_full = model.predict(X_eval)
            all_preds_mean_full = np.mean(all_preds_full)
            sample_pred_full = model.predict(X_eval.iloc[[sample_idx]])[0]
            shap_sum_full = np.sum(shap_display[sample_idx])
            shap_recon_full = base_val + shap_sum_full
            print(f"  [SHAP DEBUG] 样本总数: {len(X_display)}")
            print(f"  [SHAP DEBUG] 选中样本索引: #{sample_idx}")
            print(f"  [SHAP DEBUG] model.predict(全量) 均值: {all_preds_mean_full:.6f}")
            print(f"  [SHAP DEBUG] model.predict(选中样本): {sample_pred_full:.6f}")
            print(f"  [SHAP DEBUG] base_val: {base_val:.6f}")
            print(f"  [SHAP DEBUG] SHAP加和: {shap_sum_full:.6f}")
            print(f"  [SHAP DEBUG] 重构值: {shap_recon_full:.6f}")
            print(f"  [SHAP DEBUG] 重构 vs 直接预测 差异: {abs(shap_recon_full - sample_pred_full):.8f}")
            print(f"  [SHAP DEBUG] base_val vs 全量均值 差异: {abs(base_val - all_preds_mean_full):.8f}")
            print(f"  [SHAP DEBUG] ========== 验证结束 ==========")

            shap.waterfall_plot(shap.Explanation(
                values=shap_display[sample_idx],
                base_values=base_val,
                data=X_display.iloc[sample_idx].values,
                feature_names=list(X_display.columns)
            ), show=False)
            fig = plt.gcf()
            _fix_waterfall_labels(plt.gca(), shap_display[sample_idx], list(X_display.columns))
            ax = plt.gca()
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.4f}".rstrip('0').rstrip('.') if x != 0 else '0'))
            fig.set_size_inches(10, 6)
            fig.tight_layout()
            waterfall_path = os.path.join(output_dir, f"{safe_prop}_waterfall.png")
            fig.savefig(waterfall_path, dpi=150, bbox_inches='tight', facecolor='white')
            result['SHAP_waterfall'] = _fig_to_base64(fig)
            plt.close('all')
            print(f"  [SHAP] {prop_name}: 瀑布图已保存 (样本#{sample_idx})")

            # 3. 依赖图
            top_idx = np.argmax(np.abs(shap_display).mean(0))
            top_name = X_display.columns[top_idx]
            shap.dependence_plot(top_idx, shap_display, X_display, show=False)
            fig = plt.gcf()
            fig.set_size_inches(10, 6)
            fig.tight_layout()
            dep_path = os.path.join(output_dir, f"{safe_prop}_dependence.png")
            fig.savefig(dep_path, dpi=150, bbox_inches='tight', facecolor='white')
            result['SHAP_dependence'] = _fig_to_base64(fig)
            plt.close('all')
            print(f"  [SHAP] {prop_name}: 依赖图已保存 (TOP特征: {top_name})")

            # 导出SHAP数值数据（供图表数据导出）
            payload = _build_shap_data_payload(safe_prop, X_display, shap_display, base_val, sample_idx)
            if payload is not None:
                from .path_config import JSON_RESULT_TRAIN_ROOT
                _save_shap_data_json(payload, str(JSON_RESULT_TRAIN_ROOT / 'shap_data.json'))
            return result

        missing = [f for f in features if f not in X_full.columns]
        if missing:
            # 特征不匹配时，尝试从原始训练特征中按模型记录重新对齐。
            print(f"  [SHAP] 性质 '{prop_name}': X_full列不匹配，尝试回退到原始特征列")
            all_cols = list(train_features_df.columns[1:])
            matched = [f for f in features if f in all_cols]
            if matched:
                print(f"  [SHAP] 性质 '{prop_name}': 匹配到 {len(matched)} 个原始特征")
                X = train_features_df[matched].copy()
            else:
                print(f"  [SHAP] 性质 '{prop_name}': 特征列缺失 {missing}，跳过")
                print(f"    model expects {len(features)} features, X_full has {len(X_full.columns)} columns")
                print(f"    X_full columns (first 10): {list(X_full.columns[:10])}")
                return result
        else:
            X = X_full[features].copy()

        print(f"  [SHAP DEBUG] Before _get_shap_explainer, model_type={model_type}")
        explainer, shap_values, X_eval = _get_shap_explainer(model, X, model_type)
        print(f"  [SHAP DEBUG] After _get_shap_explainer OK")

        if hasattr(shap_values, 'values'):
            shap_array = shap_values.values
        else:
            shap_array = np.array(shap_values)

        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 0]

        X = X_eval

        X_used = X
        shap_used = shap_array
        n_used = X.shape[1]

        X_display = X_used
        shap_display = shap_used

        # 1. 全局摘要图
        shap.summary_plot(shap_display, X_display, show=False)
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        fig.tight_layout()
        summary_path = os.path.join(output_dir, f"{safe_prop}_summary.png")
        fig.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
        result['SHAP_summary'] = _fig_to_base64(fig)
        plt.close('all')
        print(f"  [SHAP] {prop_name}: 摘要图已保存 ({n_used} 个特征)")

        # 2. 局部瀑布图（随机样本，使用全部模型特征）
        sample_idx = np.random.randint(0, len(X))
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_val = float(explainer.expected_value[0])
        else:
            base_val = float(explainer.expected_value)

        # ===== 调试：验证 model.predict 是否被正确调用 =====
        print(f"  [SHAP DEBUG] ========== waterfall 模型调用验证: {prop_name} ==========")
        print(f"  [SHAP DEBUG] 模型类型: {type(model).__name__}")
        print(f"  [SHAP DEBUG] 模型信息: {model}")
        all_preds = model.predict(X)
        all_preds_mean = np.mean(all_preds)
        all_preds_std = np.std(all_preds)
        sample_pred = model.predict(X.iloc[[sample_idx]])[0]
        shap_sum = np.sum(shap_used[sample_idx])
        shap_reconstructed = base_val + shap_sum
        print(f"  [SHAP DEBUG] 样本总数: {len(X)}")
        print(f"  [SHAP DEBUG] 选中样本索引: #{sample_idx}")
        print(f"  [SHAP DEBUG] model.predict(全量X) 均值: {all_preds_mean:.6f}")
        print(f"  [SHAP DEBUG] model.predict(全量X) 标准差: {all_preds_std:.6f}")
        print(f"  [SHAP DEBUG] model.predict(选中样本): {sample_pred:.6f}")
        print(f"  [SHAP DEBUG] KernelExplainer.expected_value (base_val): {base_val:.6f}")
        print(f"  [SHAP DEBUG] SHAP值加和: {shap_sum:.6f}")
        print(f"  [SHAP DEBUG] base_val + sum(shap) 重构值: {shap_reconstructed:.6f}")
        print(f"  [SHAP DEBUG] 重构值 vs 直接预测 差异: {abs(shap_reconstructed - sample_pred):.8f}")
        print(f"  [SHAP DEBUG] base_val vs 全量预测均值 差异: {abs(base_val - all_preds_mean):.8f}")
        print(f"  [SHAP DEBUG] ========== 验证结束 ==========")

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_used[sample_idx],
                base_values=base_val,
                data=X_used.iloc[sample_idx].values,
                feature_names=list(X_used.columns)
            ),
            show=False
        )
        fig = plt.gcf()
        _fix_waterfall_labels(plt.gca(), shap_used[sample_idx], list(X_used.columns))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.4f}".rstrip('0').rstrip('.') if x != 0 else '0'))
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        waterfall_path = os.path.join(output_dir, f"{safe_prop}_waterfall.png")
        fig.savefig(waterfall_path, dpi=150, bbox_inches='tight', facecolor='white')
        result['SHAP_waterfall'] = _fig_to_base64(fig)
        plt.close('all')
        print(f"  [SHAP] {prop_name}: 瀑布图已保存 (样本#{sample_idx})")

        # 3. 特征依赖图（最重要特征）
        top_feature_idx = np.argmax(np.abs(shap_used).mean(0))
        top_feature_name = X_used.columns[top_feature_idx]

        if n_used >= 2:
            importance_order = np.argsort(np.abs(shap_used).mean(0))[::-1]
            interaction_idx = importance_order[1]
        else:
            interaction_idx = None

        shap.dependence_plot(top_feature_idx, shap_used, X_used,
                             interaction_index=interaction_idx, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        dep_path = os.path.join(output_dir, f"{safe_prop}_dependence.png")
        fig.savefig(dep_path, dpi=150, bbox_inches='tight', facecolor='white')
        result['SHAP_dependence'] = _fig_to_base64(fig)
        plt.close('all')
        print(f"  [SHAP] {prop_name}: 依赖图已保存 (TOP特征: {top_feature_name})")

        # 导出SHAP数值数据（供图表数据导出）
        payload = _build_shap_data_payload(safe_prop, X_display, shap_display, base_val, sample_idx)
        if payload is not None:
            from .path_config import JSON_RESULT_TRAIN_ROOT
            _save_shap_data_json(payload, str(JSON_RESULT_TRAIN_ROOT / 'shap_data.json'))

    except ImportError:
        print(f"  [SHAP] 性质 '{prop_name}': shap 库未安装，跳过")
    except Exception as e:
        print(f"  [SHAP] 性质 '{prop_name}': 分析失败 - {e}")

    return result


def generate_shap_all_properties(
    model_dir: str,
    property_list: List[str],
    train_features_df: pd.DataFrame,
    training_params_dir: str,
    shap_output_dir: str,
    figures_json_path: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    对所有性质生成 SHAP 分析，并集成到 figures.json（可选）

    :param model_dir: 模型保存目录
    :param property_list: 性质列表
    :param train_features_df: 原始训练集特征 DataFrame
    :param training_params_dir: 降维参数保存目录
    :param shap_output_dir: SHAP 图片输出目录
    :param figures_json_path: figures.json 路径，为 None 时跳过 base64 输出（默认关闭）
    :return: {prop_name: {图片类型: base64}}
    """
    print("\n" + "=" * 60)
    print("开始 SHAP 可解释性分析")
    print("=" * 60)

    print("将按每个 best_features 文件，使用对应模型的全部实际输入特征进行 SHAP 分析")

    os.makedirs(shap_output_dir, exist_ok=True)
    all_results = {}

    for prop in property_list:
        print(f"\n分析性质: {prop}")
        prop_results = generate_shap_for_property(
            model_dir=model_dir,
            prop_name=prop,
            train_features_df=train_features_df,
            training_params_dir=training_params_dir,
            output_dir=shap_output_dir,
        )
        if prop_results:
            all_results[prop] = prop_results

    if figures_json_path and all_results:
        _merge_shap_to_figures_json(all_results, figures_json_path)

    _cleanup_catboost_info()

    print(f"\nSHAP 分析完成，共处理 {len(all_results)} 个性质")
    print("=" * 60)
    return all_results


def _fix_waterfall_labels(ax, shap_values, feature_names):
    """修正 waterfall 图中因值过小被 shap 显示为 +0/-0 的标注"""
    import re
    name_to_shap = dict(zip(feature_names, shap_values))
    for t in ax.texts:
        txt = t.get_text().strip()
        m = re.match(r'^([+-])0(?:\.0*)?$', txt)
        if m:
            # 通过 y 坐标匹配 feature name 找到对应的 SHAP 值
            y = t.get_position()[1]
            best_name, best_val = None, 0.0
            best_dist = float('inf')
            for other in ax.texts:
                if other is t:
                    continue
                other_txt = other.get_text().strip()
                if other_txt in name_to_shap:
                    other_y = other.get_position()[1]
                    dist = abs(other_y - y)
                    if dist < best_dist:
                        best_dist = dist
                        best_name = other_txt
            if best_name and best_name in name_to_shap:
                best_val = name_to_shap[best_name]
            # 格式化：最多4位小数，不补零
            s = f"{best_val:.4f}".rstrip('0').rstrip('.')
            if s == '':
                s = '0'
            sign = '+' if best_val >= 0 else ''
            t.set_text(f"{sign}{s}")


def _build_shap_data_payload(safe_prop: str, X_display, shap_display, base_val: float, sample_idx: int):
    """构建SHAP数值数据字典（供JSON导出，用于前端复现SHAP图）"""
    try:
        X_arr = np.asarray(X_display, dtype=np.float64)
        shap_arr = np.asarray(shap_display, dtype=np.float64)
        importance = {}
        for i, feat in enumerate(X_display.columns):
            importance[feat] = round(float(np.mean(np.abs(shap_arr[:, i]))), 6)
        return {
            safe_prop: {
                'feature_names': list(X_display.columns),
                'base_value': round(float(base_val), 6),
                'waterfall_sample': int(sample_idx),
                'importance': importance,
                'samples': {
                    'data': np.round(X_arr, 6).tolist(),
                    'shap_values': np.round(shap_arr, 6).tolist(),
                },
            }
        }
    except Exception as e:
        print(f"  [SHAP] 构建SHAP数据失败: {e}")
        return None


def _save_shap_data_json(prop_data: dict, shap_data_json_path: str):
    """将SHAP数值数据合并写入 shap_data.json（供图表数据导出）"""
    try:
        existing = {}
        if os.path.exists(shap_data_json_path):
            with open(shap_data_json_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        existing.update(prop_data)
        with open(shap_data_json_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"  [SHAP] 数值数据已导出: {shap_data_json_path}")
    except Exception as e:
        print(f"  [SHAP] 导出SHAP数据失败: {e}")


def _merge_shap_to_figures_json(shap_results: Dict[str, Dict[str, str]], figures_json_path: str):
    figures_dir = os.path.dirname(figures_json_path)
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)

    existing = {}
    if os.path.exists(figures_json_path):
        with open(figures_json_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)

    for prop, img_dict in shap_results.items():
        if prop not in existing:
            existing[prop] = {}
        existing[prop].update(img_dict)

    with open(figures_json_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"SHAP 图片 base64 已写入: {figures_json_path}")
