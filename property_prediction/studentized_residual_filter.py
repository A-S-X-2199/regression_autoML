# studentized_residual_filter.py
"""
基于 Ridge / HuberRegressor 的学生化残差异常样本剔除模块

对每个目标性质独立处理：
1. 标准化特征
2. 拟合代理回归模型（Ridge 或 HuberRegressor）
3. 计算学生化残差（Studentized Residual / Externally Studentized Residual）
4. 标记 |t| > threshold 的样本为异常点
5. 将该异常样本在该性质上的目标值置 NaN（后续逐性质分割时自动排除）

原理说明：
- 学生化残差 = 残差 / (sigma * sqrt(1 - h_ii))
- 其中 h_ii 为帽子矩阵对角线（杠杆值），sigma 为删去该样本后的残差标准差估计
- HuberRegressor 对异常点天然具有鲁棒性（Huber loss），残差诊断更可靠
- 相比普通残差，学生化残差消除了杠杆值和方差差异的影响，更适合异常诊断

参考文献：
- Belsley, Kuh, Welsch (1980). Regression Diagnostics.
- Cook's Distance & Studentized Residuals.
- Huber (1964). Robust Estimation of a Location Parameter.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# 支持的代理模型类型
SUPPORTED_MODELS = {
    'ridge': Ridge,
    'huber': HuberRegressor,
}


def compute_studentized_residuals(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    model_type: str = 'ridge',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算学生化残差（Externally Studentized Residuals）

    公式：
        r_i = e_i / (sigma_(i) * sqrt(1 - h_ii))

    其中：
    - e_i: 普通残差
    - h_ii: 帽子矩阵对角线元素（杠杆值）
    - sigma_(i): 删去第 i 个样本后拟合的残差标准误差

    Args:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标值 (n_samples,)
        alpha: 正则化强度（Ridge 的 alpha，Huber 忽略）
        model_type: 代理模型类型，'ridge' 或 'huber'

    Returns:
        studentized_residuals: 学生化残差 (n_samples,)
        residuals: 普通残差 (n_samples,)
        leverages: 杠杆值 h_ii (n_samples,)
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型类型 '{model_type}'，可选: {list(SUPPORTED_MODELS.keys())}")

    n, p = X.shape

    # 1. 拟合代理回归模型
    if model_type == 'huber':
        model = HuberRegressor(max_iter=200, epsilon=1.35, alpha=0.0001)
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred

        # Huber 无闭式帽子矩阵，用线性 Ridge 近似计算杠杆值
        ridge_approx = Ridge(alpha=alpha)
        ridge_approx.fit(X, y)
        y_pred_ridge = ridge_approx.predict(X)
        residuals_for_leverage = y - y_pred_ridge

        XtX = X.T @ X
        ridge_penalty = alpha * np.eye(p)
        try:
            inv_term = np.linalg.inv(XtX + ridge_penalty)
        except np.linalg.LinAlgError:
            inv_term = np.linalg.pinv(XtX + ridge_penalty)
        H = X @ inv_term @ X.T
        leverages = np.diag(H)
    else:
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred

        # 2. 计算帽子矩阵对角元素（杠杆值）
        XtX = X.T @ X
        ridge_penalty = alpha * np.eye(p)
        try:
            inv_term = np.linalg.inv(XtX + ridge_penalty)
        except np.linalg.LinAlgError:
            inv_term = np.linalg.pinv(XtX + ridge_penalty)
        H = X @ inv_term @ X.T
        leverages = np.diag(H)

    # 裁剪杠杆值到 [0, 1)，防止 sqrt(1 - h_ii) 为负数或虚数
    leverages = np.clip(leverages, 0.0, 1.0 - 1e-12)

    # 3. 计算学生化残差（外部学生化）
    mse = np.sum(residuals ** 2) / (n - p)  # 均方误差
    studentized = np.zeros(n)

    for i in range(n):
        h_ii = leverages[i]
        sigma_i_sq = ((n - p) * mse - residuals[i] ** 2 / (1 - h_ii)) / (n - p - 1)
        sigma_i_sq = max(sigma_i_sq, 1e-12)  # 防零
        sigma_i = np.sqrt(sigma_i_sq)

        studentized[i] = residuals[i] / (sigma_i * np.sqrt(1 - h_ii))

    return studentized, residuals, leverages


def plot_outlier_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    studentized: np.ndarray,
    leverages: np.ndarray,
    outlier_mask: np.ndarray,
    prop_name: str,
    threshold: float,
    output_dir: Path,
    sample_ids: np.ndarray = None,
):
    """
    生成异常样本剔除的诊断可视化图表。

    包含四张子图：
    1. 预测 vs 实测散点图（正常蓝、异常红，含 y=x 参考线和 R²）
    2. 学生化残差分布直方图（含 ±threshold 阈值线）
    3. 学生化残差 vs 预测值散点图（含阈值线，标注样本编号）
    4. 学生化残差 vs 杠杆值散点图（含阈值线）
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    safe_prop = prop_name.replace('/', '_').replace('\\', '_')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'异常样本诊断 — {prop_name}\n(阈值: |t| > {threshold})', 
                 fontsize=18, fontweight='bold', y=0.98)

    normal_mask = ~outlier_mask

    # ===== 子图1: 预测 vs 实测 =====
    ax1 = axes[0, 0]
    ax1.scatter(y_true[normal_mask], y_pred[normal_mask], c='#2c7bb6', alpha=0.7,
                edgecolors='white', linewidth=0.5, s=80, label=f'正常样本 ({normal_mask.sum()})')
    if outlier_mask.sum() > 0:
        ax1.scatter(y_true[outlier_mask], y_pred[outlier_mask], c='#d7191c', alpha=0.9,
                    edgecolors='darkred', linewidth=1.5, s=120, marker='X',
                    label=f'异常样本 ({outlier_mask.sum()})')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    pad = (lims[1] - lims[0]) * 0.05
    ax1.plot([lims[0] - pad, lims[1] + pad], [lims[0] - pad, lims[1] + pad],
             'k--', alpha=0.4, linewidth=1.5, label='y=x')
    r2 = r2_score(y_true, y_pred)
    ax1.set_xlabel('实测值', fontsize=14)
    ax1.set_ylabel('预测值', fontsize=14)
    ax1.set_title(f'预测 vs 实测  (R2 = {r2:.4f},  n = {len(y_true)})', fontsize=15)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)

    # ===== 子图2: 学生化残差直方图 =====
    ax2 = axes[0, 1]
    counts, bins, patches = ax2.hist(studentized, bins=min(30, len(y_true) // 3), 
                                      color='#2c7bb6', alpha=0.7, edgecolor='white', linewidth=0.8)
    for i, (cnt, edge) in enumerate(zip(counts, bins[:-1])):
        if abs(edge) >= threshold or abs(bins[i + 1]) >= threshold:
            patches[i].set_facecolor('#d7191c')
            patches[i].set_alpha(0.6)
    ax2.axvline(threshold, color='#d7191c', linestyle='--', linewidth=2, label=f'+{threshold}σ')
    ax2.axvline(-threshold, color='#d7191c', linestyle='--', linewidth=2, label=f'-{threshold}σ')
    ax2.set_xlabel('学生化残差 t', fontsize=14)
    ax2.set_ylabel('频数', fontsize=14)
    ax2.set_title(f'学生化残差分布  (超出阈值: {outlier_mask.sum()} 个)', fontsize=15)
    ax2.legend(fontsize=12)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== 子图3: 学生化残差 vs 预测值 =====
    ax3 = axes[1, 0]
    ax3.scatter(y_pred[normal_mask], studentized[normal_mask], c='#2c7bb6', alpha=0.7,
                edgecolors='white', linewidth=0.5, s=70)
    if outlier_mask.sum() > 0:
        ax3.scatter(y_pred[outlier_mask], studentized[outlier_mask], c='#d7191c', alpha=0.9,
                    edgecolors='darkred', linewidth=1, s=100, marker='X')
        if sample_ids is not None:
            outlier_samples = sample_ids[outlier_mask]
            for i, rid in zip(np.where(outlier_mask)[0], outlier_samples):
                ax3.annotate(str(rid), (y_pred[i], studentized[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8,
                             fontweight='bold')
    ax3.axhline(threshold, color='#d7191c', linestyle='--', linewidth=1.5)
    ax3.axhline(-threshold, color='#d7191c', linestyle='--', linewidth=1.5)
    ax3.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax3.set_xlabel('预测值', fontsize=14)
    ax3.set_ylabel('学生化残差 t', fontsize=14)
    ax3.set_title('学生化残差 vs 预测值', fontsize=15)
    ax3.tick_params(labelsize=12)
    ax3.grid(True, alpha=0.3)

    # ===== 子图4: 学生化残差 vs 杠杆值 =====
    ax4 = axes[1, 1]
    ax4.scatter(leverages[normal_mask], studentized[normal_mask], c='#2c7bb6', alpha=0.7,
                edgecolors='white', linewidth=0.5, s=70)
    if outlier_mask.sum() > 0:
        ax4.scatter(leverages[outlier_mask], studentized[outlier_mask], c='#d7191c', alpha=0.9,
                    edgecolors='darkred', linewidth=1, s=100, marker='X')
    ax4.axhline(threshold, color='#d7191c', linestyle='--', linewidth=1.5)
    ax4.axhline(-threshold, color='#d7191c', linestyle='--', linewidth=1.5)
    ax4.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax4.set_xlabel('杠杆值 h_ii', fontsize=14)
    ax4.set_ylabel('学生化残差 t', fontsize=14)
    ax4.set_title('学生化残差 vs 杠杆值', fontsize=15)
    ax4.tick_params(labelsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = output_dir / f'{safe_prop}_outlier_diagnostics.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"    诊断图已保存: {fig_path}")

    return str(fig_path)


def export_outlier_samples_to_excel(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    property_names: List[str],
    outlier_info: Dict[str, Dict],
    output_dir: Path,
    id_col: str,
):
    """
    将各目标的异常样本及其全部特征导出到一个 Excel 文件。
    """
    if not outlier_info:
        return

    excel_path = output_dir / 'outlier_samples.xlsx'
    
    with pd.ExcelWriter(str(excel_path), engine='openpyxl') as writer:
        # 汇总 sheet
        summary_rows = []
        for prop in property_names:
            info = outlier_info.get(prop, {})
            if info.get('outlier_count', 0) > 0:
                summary_rows.append({
                    '目标': prop,
                    '总样本数': info.get('total_samples', 0),
                    '有效样本数': info.get('valid_samples', 0),
                    '异常样本数': info.get('outlier_count', 0),
                    '异常比例': f"{info.get('outlier_ratio', 0):.2%}",
                    '学生化残差max': info.get('studentized_max', 0),
                })
        
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='汇总', index=False)
        
        # 每个目标单独一个 sheet
        for prop in property_names:
            info = outlier_info.get(prop, {})
            if info.get('outlier_count', 0) == 0:
                continue
            
            safe_prop = prop.replace('/', '_').replace('\\', '_')[:31]  # Excel sheet name limit
            
            # 使用诊断阶段记录的精确索引，避免把原本缺失目标的样本误报为异常。
            outlier_indices = info.get('outlier_indices', [])
            if not outlier_indices:
                continue
            outlier_df = features_df.loc[outlier_indices].copy()
            outlier_df.insert(1, '目标名称', prop)
            outlier_df.insert(2, '目标实测值', info.get('outlier_values', []))
            
            outlier_df.to_excel(writer, sheet_name=safe_prop, index=False)
    
    print(f"  异常样本 Excel 已导出: {excel_path}")
    return str(excel_path)


def filter_outliers_per_property(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    property_names: List[str],
    std_threshold: float = 2.0,
    outlier_model: str = 'ridge',
    ridge_alpha: float = 1.0,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    """
    对每个目标独立执行基于代理回归 + 学生化残差的异常样本剔除。

    对每个性质：
    1. 提取该性质有效样本（目标值非 NaN）
    2. 对特征做标准化，特征数 > 样本数时先 PCA 降维
    3. 拟合代理回归模型（Ridge 或 HuberRegressor）
    4. 计算学生化残差
    5. 将 |studentized_residual| > std_threshold 的样本在该性质上的目标值置 NaN

    注意：不同性质的异常样本可能不同，
    异常样本在该性质的目标值被置 NaN 后，
    后续逐性质分割时会自动排除。

    Args:
        features_df: 特征 DataFrame（第一列为样本编号）
        targets_df: 目标 DataFrame（第一列为样本编号）
        property_names: 要处理的目标列表
        std_threshold: 学生化残差阈值（默认 2.0，即 2 倍标准差）
        outlier_model: 代理模型类型，'ridge' 或 'huber'（默认 'ridge'）
        ridge_alpha: Ridge 正则化强度（huber 模式下忽略）
        random_state: 随机种子

    Returns:
        features_df: 处理后的特征 DataFrame（不变，仅目标值被修改）
        targets_df: 处理后的目标 DataFrame（异常样本目标值置 NaN）
        outlier_info: 每个性质的异常信息 dict
    """
    if outlier_model not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的异常检测模型 '{outlier_model}'，可选: {list(SUPPORTED_MODELS.keys())}")

    model_name = {'ridge': 'Ridge', 'huber': 'HuberRegressor'}[outlier_model]
    print("\n" + "=" * 60)
    print(f"学生化残差异常样本剔除（{model_name} + Externally Studentized Residuals）")
    print(f"  阈值: |t| > {std_threshold}")
    if outlier_model == 'ridge':
        print(f"  Ridge alpha: {ridge_alpha}")
    print("=" * 60)

    if features_df.shape[0] == 0 or targets_df.shape[0] == 0:
        print("警告: 数据为空，跳过异常剔除")
        return features_df, targets_df, {}

    id_col = features_df.columns[0]
    targets_df = targets_df.copy()
    outlier_info = {}
    total_outliers = 0

    for prop in property_names:
        if prop not in targets_df.columns:
            print(f"\n  性质 '{prop}' 不在目标数据中，跳过")
            continue

        # 性质处理日志头：先打印标题，便于与后续PCA/异常信息对应
        print(f"\n性质 '{prop}':")

        # 1. 提取该性质有效样本
        y_series = targets_df[prop].copy()
        valid_mask = y_series.notna()

        if valid_mask.sum() < 10:
            print(f"\n  性质 '{prop}': 有效样本仅 {valid_mask.sum()} 个 (<10)，跳过异常剔除")
            outlier_info[prop] = {
                'total_samples': len(y_series),
                'valid_samples': int(valid_mask.sum()),
                'outlier_count': 0,
                'skipped': True,
                'skipped_reason': '样本数不足 (<10)',
            }
            continue

        valid_indices = y_series[valid_mask].index
        y_valid = y_series[valid_mask].values.astype(np.float64)

        # 特征对齐：用 features_df 中 valid_mask 对应的行
        X_valid_df = features_df.loc[valid_indices].copy()
        # 异常样本诊断直接使用当前目标对应样本的全部可用特征。
        feature_cols = [c for c in X_valid_df.columns if c != id_col]
        X_valid = X_valid_df[feature_cols].values.astype(np.float64)

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)

        # 当特征数多于样本数时，先 PCA 降维确保回归可计算
        n, p = X_scaled.shape
        pca_n_components = None
        if n < p + 2:
            # 遍历降维比例列表筛选 PCA 维数，步长由全部可用特征数确定，
            # 与模型优化器 _build_reduction_ratio_list 保持一致；方差解释度>=0.8 即截至
            back_count = p
            if back_count <= 50:
                step = 0.10
            elif back_count <= 200:
                step = 0.05
            else:
                step = 0.03
            ratios = []
            r = 0.05
            while r <= 0.95 + 1e-9:
                ratios.append(round(r, 2))
                r += step

            # 最大可取维数：样本充足性约束，确保学生化残差计算 n-p-1>0
            max_components = max(1, min(n // 2, p))
            chosen = None
            chosen_explained = 0.0
            for ratio in ratios:
                n_comp = max(1, min(int(round(ratio * p)), max_components))
                pca = PCA(n_components=n_comp, random_state=random_state)
                pca.fit(X_scaled)
                explained = pca.explained_variance_ratio_.sum()
                chosen, chosen_explained = n_comp, explained
                if explained >= 0.8:
                    break
            pca_n_components = chosen if chosen is not None else max_components
            pca = PCA(n_components=pca_n_components, random_state=random_state)
            X_scaled = pca.fit_transform(X_scaled)
            print(f"  PCA降维筛选: {p} → {pca_n_components} "
                  f"(n={n}, 累计解释方差比: {chosen_explained:.3f}, 截止阈值≥0.8)")

        # 计算学生化残差
        try:
            studentized, residuals, leverages = compute_studentized_residuals(
                X_scaled, y_valid, alpha=ridge_alpha, model_type=outlier_model
            )
        except Exception as e:
            print(f"\n  性质 '{prop}': 计算学生化残差失败 ({e})，跳过")
            outlier_info[prop] = {
                'total_samples': len(y_series),
                'valid_samples': int(valid_mask.sum()),
                'outlier_count': 0,
                'skipped': True,
                'skipped_reason': f'计算失败: {str(e)[:60]}',
            }
            continue

        # 4. 标记异常样本
        outlier_mask = np.abs(studentized) > std_threshold
        outlier_count = int(outlier_mask.sum())

        outlier_indices_in_valid = valid_indices[outlier_mask]
        outlier_values = y_series.loc[outlier_indices_in_valid].astype(float).tolist()

        # 5. 将异常样本在该目标上的目标值置 NaN
        if outlier_count > 0:
            targets_df.loc[outlier_indices_in_valid, prop] = np.nan
            total_outliers += outlier_count

        # 生成诊断可视化
        if output_dir is not None:
            try:
                y_pred = y_valid - residuals
                sample_ids = features_df.loc[valid_indices, id_col].values
                plot_outlier_diagnostics(
                    y_true=y_valid,
                    y_pred=y_pred,
                    studentized=studentized,
                    leverages=leverages,
                    outlier_mask=outlier_mask,
                    prop_name=prop,
                    threshold=std_threshold,
                    output_dir=output_dir,
                    sample_ids=sample_ids,
                )
            except Exception as e_plot:
                print(f"    诊断图生成失败: {e_plot}")

        # 记录信息
        outlier_info[prop] = {
            'total_samples': len(y_series),
            'valid_samples': int(valid_mask.sum()),
            'outlier_count': outlier_count,
            'outlier_ratio': round(outlier_count / max(valid_mask.sum(), 1), 4),
            'skipped': False,
            'skipped_reason': '',
            'pca_reduced': pca_n_components is not None,
            'pca_n_components': pca_n_components,
            'studentized_max': float(np.max(np.abs(studentized))),
            'studentized_status': _describe_studentized(studentized, outlier_mask, std_threshold),
            'outlier_indices': outlier_indices_in_valid.tolist(),
            'outlier_values': outlier_values,
        }

        sample_ids = features_df.loc[valid_indices, id_col].values
        outlier_samples = [str(sample_ids[i]) for i in range(len(sample_ids)) if outlier_mask[i]]

        print(f"    总样本: {len(y_series)}, 有效: {valid_mask.sum()}, "
              f"异常: {outlier_count}")
        if outlier_count > 0:
            print(f"    异常样本: {', '.join(outlier_samples)}")
            print(f"    学生化残差范围: [{np.min(studentized):.3f}, {np.max(studentized):.3f}]")
            print(f"    异常阈值线: ±{std_threshold}")

    print(f"\n  总计剔除异常样本: {total_outliers} 个（跨所有目标，同一样本可能被多次标记）")
    print("=" * 60)

    # 汇总导出异常样本到 Excel
    if output_dir is not None and total_outliers > 0:
        try:
            export_outlier_samples_to_excel(
                features_df=features_df,
                targets_df=targets_df,
                property_names=property_names,
                outlier_info=outlier_info,
                output_dir=output_dir,
                id_col=id_col,
            )
        except Exception as e_excel:
            print(f"  异常样本 Excel 导出失败: {e_excel}")

    return features_df, targets_df, outlier_info


def _describe_studentized(
    studentized: np.ndarray,
    outlier_mask: np.ndarray,
    threshold: float,
) -> Dict:
    """生成学生化残差的描述性统计"""
    abs_s = np.abs(studentized)
    return {
        'mean': float(np.mean(studentized)),
        'std': float(np.std(studentized)),
        'min': float(np.min(studentized)),
        'max': float(np.max(studentized)),
        'abs_max': float(np.max(abs_s)),
        'threshold': threshold,
        'above_1sigma': int(np.sum(abs_s > 1.0)),
        'above_2sigma': int(np.sum(abs_s > 2.0)),
        'above_3sigma': int(np.sum(abs_s > 3.0)),
        'outliers_found': int(np.sum(outlier_mask)),
    }
