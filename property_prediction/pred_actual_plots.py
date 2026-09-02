import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import os, re
import json

def load_best_models_info(properties_list, best_models_dir="best_models"):
    """从保存的模型信息文件中加载 best_models_info"""
    best_models_info = {}
    save_dir = best_models_dir
    
    for prop in properties_list:
        # 清理性质名称,与保存时一致
        safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")
        info_file = f"{save_dir}/best_model_info_{safe_prop}.json"
        
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                best_models_info[prop] = model_info
                print(f"加载性质 {prop} 的模型信息: {model_info.get('best_model', '未知')}")
            except Exception as e:
                print(f"加载 {info_file} 失败: {e}")
        else:
            print(f"警告: 性质 {prop} 的模型信息文件不存在: {info_file}")
    
    return best_models_info

def plot_pred_actual_with_train_test(
    train_pred_file,    # 训练集预测结果Excel路径 (best_model_perform.xlsx)
    test_pred_file,     # 测试集预测结果Excel路径
    test_pred_sheet,    # 测试集预测结果sheet名
    test_actual_file,   # 测试集实际值Excel路径
    test_actual_sheet,  # 测试集实际值sheet名
    output_img_prefix,  # 输出图片路径前缀
    property_list=None, # 指定要处理的性质列表
    best_models_dir="best_models",  # 最优模型目录
    figsize_scatter=(18, 14),   # 增大的散点图大小
    figsize_bar=(20, 8),        # 增大的条形图大小
    cols_per_row=2,     # 每行子图数
    scale_factor=1.0,    # 坐标轴缩放因子
    random_seed=42,
    metrics_json_dir="test_metrics_json",  # 英文JSON保存目录
    train_data_dict: dict = None,  # 训练集数据字典 {prop: DataFrame}
    test_data_dict = None,  # 测试集预测DataFrame,替代test_pred_file
    test_actual_df = None,  # 测试集实际值DataFrame,替代test_actual_file
    outlier_n: int = None,  # 异常点标记数,不为None且≥0时标记测试集最大误差的n个点
    test_ids_per_prop: dict = None,  # 按性质测试集划分 {prop: [样本id,...]},提供时测试集数据仅保留该性质自己的测试集样本
):
    """生成多个性质的预测-实际对比散点图(包含训练集和测试集)和性能指标条形图
    """
    
    # 读取训练集预测数据
    train_data = {}
    if train_data_dict is not None:
        for prop, val in train_data_dict.items():
            if property_list is not None and prop not in property_list:
                continue
            if isinstance(val, dict) and 'predictions' in val:
                df = pd.DataFrame(val['predictions'])
            elif isinstance(val, pd.DataFrame):
                df = val
            else:
                df = pd.DataFrame(val)
            exp_cols = [col for col in df.columns if col.endswith('_exp')]
            pred_cols = [col for col in df.columns if col.endswith('_pred')]
            if exp_cols and pred_cols:
                train_data[prop] = {
                    'actual': df[exp_cols[0]].values,
                    'pred': df[pred_cols[0]].values,
                    'source': '训练集'
                }
    else:
        print(f"警告: 未提供训练集数据字典 train_data_dict")
    
    # 读取测试集预测数据
    if test_data_dict is not None:
        if isinstance(test_data_dict, pd.DataFrame):
            test_pred_df = test_data_dict
        else:
            test_pred_df = pd.DataFrame(test_data_dict)
    else:
        raise ValueError("未提供测试集预测数据 test_data_dict")
    # 预测列 = 排除元数据列后的所有列（预测结果表元数据列为：样本编号/数据集类型/原始编号/样本标识）
    _meta_cols = ('样本编号', '数据集类型', '原始编号', '样本标识')
    test_pred_cols = [col for col in test_pred_df.columns if col not in _meta_cols]
    
    # 如果指定了property_list,筛选测试集预测数据列
    if property_list is not None:
        test_pred_cols = [col for col in test_pred_cols if col in property_list]
    
    if not test_pred_cols:
        raise ValueError("在测试集预测数据中没有找到指定的性质")
        
    test_pred_data = test_pred_df[test_pred_cols].values
    
    # 读取测试集实际数据
    if test_actual_df is not None:
        _test_actual_df = test_actual_df
    else:
        raise ValueError("未提供测试集实际数据 test_actual_df")
    test_actual_candidate_cols = _test_actual_df.iloc[:, 1:].columns.tolist()
    test_actual_matched_cols = [col for col in test_pred_cols if col in test_actual_candidate_cols]
    
    if not test_actual_matched_cols:
        raise ValueError("测试集实际数据表中没有找到与预测数据表匹配的字段")
    
    test_actual_data = _test_actual_df[test_actual_matched_cols].values
    
    # 准备测试集数据
    # 提取样本编号（用于异常点标注）
    test_recipe_ids = test_pred_df.iloc[:, 0].values if test_pred_df.shape[1] >= 1 else None
    _actual_df_id_col = _test_actual_df.columns[0]
    for i, prop in enumerate(test_actual_matched_cols):
        # 若提供按性质测试集划分,仅保留该性质自己的测试集样本(pred/actual按id对齐)
        _prop_ids = test_ids_per_prop.get(prop) if test_ids_per_prop else None
        if _prop_ids:
            _str_ids = [str(x) for x in _prop_ids]
            _pred_mask = np.isin([str(x) for x in test_recipe_ids], _str_ids)
            _pred_sel_ids = np.asarray(test_recipe_ids)[_pred_mask]
            _t_pred = np.asarray(test_pred_data[:, i])[_pred_mask]
            # actual 按 id 对齐到 pred 的顺序
            _id_to_actual = dict(zip(
                [str(x) for x in _test_actual_df[_actual_df_id_col].values],
                np.asarray(test_actual_data[:, i])
            ))
            _t_actual = np.array(
                [_id_to_actual.get(str(pid), np.nan) for pid in _pred_sel_ids],
                dtype=np.float64
            )
            _t_recipe_ids = _pred_sel_ids
        else:
            _t_actual = test_actual_data[:, i]
            _t_pred = test_pred_data[:, i]
            _t_recipe_ids = test_recipe_ids

        test_data = {
            'actual': _t_actual,
            'pred': _t_pred,
            'source': '测试集'
        }
        # 附加样本编号
        if _t_recipe_ids is not None:
            test_data['recipe_ids'] = _t_recipe_ids
        
        # 如果训练集中已有该性质,合并数据
        if prop in train_data:
            train_test_data = {
                '训练集': train_data[prop],
                '测试集': test_data
            }
            train_data[prop] = train_test_data
        else:
            train_data[prop] = {'测试集': test_data}
    
    # 计算性能指标,同时保留中文兼容字段和英文标准字段
    metrics = {}
    for prop, data_dict in train_data.items():
        # 如果指定了property_list,只处理列表中的性质
        if property_list is not None and prop not in property_list:
            continue
            
        metrics[prop] = {}
        
        if isinstance(data_dict, dict):
            for source, data in data_dict.items():
                if isinstance(data, dict) and 'actual' in data and 'pred' in data:
                    y_true = data['actual']
                    y_pred = data['pred']
                    
                    # 过滤NaN值
                    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                    y_true_clean = y_true[mask]
                    y_pred_clean = y_pred[mask]
                    
                    if len(y_true_clean) > 0:
                        r2 = r2_score(y_true_clean, y_pred_clean)
                        mae = mean_absolute_error(y_true_clean, y_pred_clean)
                        mse = mean_squared_error(y_true_clean, y_pred_clean)
                        rmse = np.sqrt(mse)
                        mape = calculate_mape(y_true_clean, y_pred_clean)
                        
                        print(f"性质: {prop}, 数据源: {source}, R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")
                        
                        # 计算P值和Pearson相关系数
                        if len(y_true_clean) > 2:
                            pearson_corr, pearson_p_value = stats.pearsonr(y_true_clean, y_pred_clean)
                        else:
                            pearson_corr, pearson_p_value = np.nan, np.nan
                        
                        # 1. 中文兼容版指标：用于原有可视化、Excel,完全不变
                        cn_metrics = {
                            'R2': r2,
                            'MAE': mae,
                            'MSE': mse,
                            'RMSE': rmse,
                            'MAPE': mape,
                            'P值': pearson_p_value,
                            'Pearson相关系数': pearson_corr,
                            '样本数': len(y_true_clean)
                        }
                        
                        # 2. 纯英文标准版指标,用于JSON输出
                        en_metrics_output = {
                            'property': prop,
                            'pearson_corr': pearson_corr,
                            'pearson_p_value': pearson_p_value,
                            'mape': mape,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'sample count': len(y_true_clean)
                        }
                        
                        metrics[prop][source] = {
                            "cn": cn_metrics,
                            "en": en_metrics_output
                        }

    # 遍历所有性质,分别保存英文、中文JSON
    for prop in metrics:
        if '测试集' not in metrics[prop]:
            print(f"警告: 性质 {prop} 无测试集评估指标,跳过JSON保存")
            continue
        
        # 提取对应版本指标
        test_en = metrics[prop]['测试集']['en'].copy()
        
        # 处理numpy nan/inf,转为JSON支持的None
        def process_nan_inf(metric_dict):
            return {
                k: (v if not (isinstance(v, (int, float, np.number)) and (np.isnan(v) or np.isinf(v))) else None) 
                for k, v in metric_dict.items()
            }
        test_en_processed = process_nan_inf(test_en)
        en_json_path = os.path.join(metrics_json_dir, f"{prop}_test_metrics.json")
        with open(en_json_path, 'w', encoding='utf-8') as f:
            json.dump(test_en_processed, f, ensure_ascii=False, indent=4, default=str)
        print(f"JSON已保存：{en_json_path}")
    best_models_info = load_best_models_info(list(train_data.keys()), best_models_dir)

    plot_scatter_with_train_test(train_data, metrics, output_img_prefix, property_list, 
                               best_models_info, figsize_scatter, scale_factor=scale_factor,
                               outlier_n=outlier_n)
    
    plot_bar_metrics(metrics, f"{output_img_prefix}_bar.png", random_seed, figsize_bar, property_list)

    # 导出预测-实际图表数据到JSON（散点图/簇状条形图/指标条形图数据，供前端复现）
    try:
        data_export = {}
        for prop, data_dict in train_data.items():
            if property_list is not None and prop not in property_list:
                continue
            entry = {'metrics': {}}
            for source in ['训练集', '测试集']:
                if source in data_dict and isinstance(data_dict[source], dict):
                    d = data_dict[source]
                    actual_arr = np.asarray(d.get('actual', []), dtype=np.float64).ravel()
                    pred_arr = np.asarray(d.get('pred', []), dtype=np.float64).ravel()
                    if 'recipe_ids' in d and d['recipe_ids'] is not None:
                        samples = [str(x) for x in np.asarray(d['recipe_ids']).ravel()]
                    else:
                        prefix = 'Train' if source == '训练集' else 'Test'
                        samples = [f"{prefix}_{i+1}" for i in range(len(actual_arr))]
                    entry[source] = {
                        'samples': samples,
                        'actual': _np_to_json(actual_arr),
                        'pred': _np_to_json(pred_arr),
                    }
                if source in metrics.get(prop, {}):
                    entry['metrics'][source] = _np_to_json(metrics[prop][source].get('cn', {}))
            data_export[prop] = entry
        if data_export:
            out_path = os.path.join(metrics_json_dir, 'pred_actual_data.json')
            existing = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                except Exception:
                    existing = {}
            existing.update(data_export)
            os.makedirs(metrics_json_dir, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2, default=str)
            print(f"预测-实际图表数据已导出: {out_path}")
    except Exception as e:
        print(f"导出预测-实际图表数据失败: {e}")

    return metrics


def _np_to_json(obj):
    """numpy数组/数值转JSON可序列化对象，NaN/Inf转None"""
    if isinstance(obj, dict):
        return {k: _np_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _np_to_json(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float, int)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    return obj

def calculate_mape(y_true, y_pred, eps=1e-6):
    """计算平均绝对百分比误差"""
    y_true = np.where(abs(y_true) < eps, eps, y_true)
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

def plot_scatter_with_train_test(data_dict, metrics, output_img_prefix, property_list=None, 
                               best_models_info=None, figsize=(18, 14), scale_factor=1.0,
                               outlier_n: int = None):
    """绘制包含训练集和测试集的散点图,每个性质单独一张图,显示最佳模型名称。
    当 outlier_n 不为 None 且 >=0 时，对测试集标记最大误差的 n 个点（红色+样本号），
    并显示剔除这些点后的 R2 和 MAPE。
    """
    
    # 中文字体设置和增大全局字号
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    # ["SimHei", "SimSun", "Noto Sans SC", "WenQuanYi Zen Hei", "DejaVu Sans", "Helvetica"]
    # ["SimSun","DejaVu Sans"]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': 16,           # 全局字号
        'axes.titlesize': 28,      # 标题字号
        'axes.labelsize': 24,      # 坐标轴标签字号
        'legend.fontsize': 20,     # 图例字号
        'xtick.labelsize': 18,     # X轴刻度字号
        'ytick.labelsize': 18,     # Y轴刻度字号
        'figure.titlesize': 30     # 图形标题字号
    })
    
    # 创建散点图文件夹
    scatter_dir = f"{output_img_prefix}_scatter"
    os.makedirs(scatter_dir, exist_ok=True)
    
    # 如果指定了property_list,只处理列表中的性质
    if property_list is not None:
        properties = [prop for prop in data_dict.keys() if prop in property_list]
    else:
        properties = list(data_dict.keys())
    
    # 颜色设置
    colors = {'训练集': '#2E86AB', '测试集': '#A23B72'}
    
    for prop in properties:
        fig, ax = plt.subplots(figsize=figsize)
        prop_data = data_dict[prop]
        
        # 确保 prop_data 是字典
        if not isinstance(prop_data, dict):
            print(f"警告: 性质 {prop} 的数据不是字典格式,跳过绘图")
            continue
        
        all_actual = []
        all_pred = []
        
        # 绘制训练集数据
        if '训练集' in prop_data:
            train_data = prop_data['训练集']
            if isinstance(train_data, dict) and 'actual' in train_data and 'pred' in train_data:
                train_actual = train_data['actual']
                train_pred = train_data['pred']
                mask = ~(np.isnan(train_actual) | np.isnan(train_pred))
                if np.any(mask):
                    ax.scatter(train_actual[mask], train_pred[mask], 
                              alpha=0.7, s=80, color=colors['训练集'], label='训练集')
                    all_actual.extend(train_actual[mask])
                    all_pred.extend(train_pred[mask])
        
        # 绘制测试集数据
        _test_outlier_data = None  # 异常点标注数据
        if '测试集' in prop_data:
            test_data = prop_data['测试集']
            if isinstance(test_data, dict) and 'actual' in test_data and 'pred' in test_data:
                test_actual = test_data['actual']
                test_pred = test_data['pred']
                recipe_ids = test_data.get('recipe_ids', None)
                mask = ~(np.isnan(test_actual) | np.isnan(test_pred))
                if np.any(mask):
                    t_actual = np.asarray(test_actual)[mask]
                    t_pred = np.asarray(test_pred)[mask]
                    t_recipes = np.asarray(recipe_ids)[mask] if recipe_ids is not None else None

                    # 异常点逻辑
                    outlier_mask = np.zeros(len(t_actual), dtype=bool)
                    if outlier_n is not None and outlier_n > 0 and len(t_actual) > outlier_n + 1:
                        abs_err = np.abs(t_actual - t_pred)
                        outlier_indices = np.argsort(abs_err)[-outlier_n:]
                        outlier_mask[outlier_indices] = True

                        # 剔除异常点后指标
                        keep = ~outlier_mask
                        adj_r2 = r2_score(t_actual[keep], t_pred[keep])
                        adj_mape = calculate_mape(t_actual[keep], t_pred[keep])
                        outlier_recipe_ids = t_recipes[outlier_indices] if t_recipes is not None else None

                        # 正常点(蓝色)
                        normal_mask = ~outlier_mask
                        ax.scatter(t_actual[normal_mask], t_pred[normal_mask],
                                  alpha=0.7, s=80, color=colors['测试集'], label='测试集')
                        # 异常点(红色)
                        ax.scatter(t_actual[outlier_mask], t_pred[outlier_mask],
                                  alpha=0.9, s=150, color='red', edgecolors='darkred',
                                  linewidths=1.5, marker='o', zorder=5, label=f'Top{outlier_n}误差')
                        all_actual.extend(t_actual)
                        all_pred.extend(t_pred)

                        # 异常点数据暂存,待理想线绘制后再标注
                        _test_outlier_data = (t_actual, t_pred, outlier_indices,
                                             outlier_recipe_ids, adj_r2, adj_mape, outlier_n)
                    else:
                        _test_outlier_data = None
                        ax.scatter(t_actual, t_pred,
                                  alpha=0.7, s=80, color=colors['测试集'], label='测试集')
                        all_actual.extend(t_actual)
                        all_pred.extend(t_pred)
        
        if all_actual and all_pred:
            # 计算整体范围
            min_val = min(min(all_actual), min(all_pred))
            max_val = max(max(all_actual), max(all_pred))
            
            # 应用缩放因子
            if scale_factor != 1.0:
                center = (min_val + max_val) / 2
                half_range = (max_val - min_val) / 2
                scaled_half_range = half_range * scale_factor
                min_val_scaled = center - scaled_half_range
                max_val_scaled = center + scaled_half_range
                
                ax.set_xlim(min_val_scaled, max_val_scaled)
                ax.set_ylim(min_val_scaled, max_val_scaled)
                
                # 绘制理想线
                ax.plot([min_val_scaled, max_val_scaled], [min_val_scaled, max_val_scaled], 
                       'k--', linewidth=3, alpha=0.8, label='理想线')
            else:
                ax.plot([min_val, max_val], [min_val, max_val], 
                       'k--', linewidth=3, alpha=0.8, label='理想线')
            
            # 添加性能指标文本和最佳模型名称
            text_lines = []
            
            # 添加最佳模型信息
            if best_models_info and prop in best_models_info:
                best_model = best_models_info[prop].get('best_model', '未知')
                text_lines.append(f'最佳模型: {best_model}')
            
            for source in ['训练集', '测试集']:
                if source in metrics.get(prop, {}):
                    # 修复：提取cn模式指标
                    m = metrics[prop][source]["cn"]
                    text_lines.append(f'{source}:')
                    text_lines.append(f'  R² = {m["R2"]:.4f}')
                    text_lines.append(f'  MSE = {m["MSE"]:.4f}')
                    text_lines.append(f'  RMSE = {m["RMSE"]:.4f}')
                    text_lines.append(f'  MAE = {m["MAE"]:.4f}')
                    text_lines.append(f'  MAPE = {m["MAPE"]:.2f}%')
                    text_lines.append(f'  Pearson = {m["Pearson相关系数"]:.4f}')
                    text_lines.append('')
            
            if text_lines:
                ax.text(0.05, 0.95, '\n'.join(text_lines),
                        transform=ax.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        verticalalignment='top', 
                        fontsize=18,
                        linespacing=1.5)

        # ===== 异常点标注：红点旁样本号 + 剔除后指标 =====
        if _test_outlier_data is not None:
            t_actual, t_pred, outlier_indices, outlier_recipe_ids, adj_r2, adj_mape, o_n = _test_outlier_data
            for i, idx in enumerate(outlier_indices):
                rid = outlier_recipe_ids[i] if outlier_recipe_ids is not None else '?'
                ax.annotate(str(rid), (t_actual[idx], t_pred[idx]),
                            fontsize=8, color='darkred', fontweight='bold',
                            ha='center', va='bottom',
                            xytext=(0, 10), textcoords='offset points')
            adj_text = (
                f"剔除Top{o_n}误差后:\n"
                f"  R² = {adj_r2:.4f}\n"
                f"  MAPE = {adj_mape:.2f}%"
            )
            ax.text(0.95, 0.5, adj_text,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='orange'),
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=18)
        
        # 设置标题 - 包含最佳模型名称
        if best_models_info and prop in best_models_info:
            best_model = best_models_info[prop].get('best_model', '未知')
            title = f'{prop} - 预测 vs 实际 (最佳模型: {best_model})'
        else:
            title = f'{prop} - 预测 vs 实际'
            
        ax.set_title(title, fontsize=28, fontweight='bold', pad=20)
        ax.set_xlabel('实际值', fontsize=24, labelpad=15)
        ax.set_ylabel('预测值', fontsize=24, labelpad=15)
        ax.legend(fontsize=20, loc='lower right')
        ax.grid(True, alpha=0.3, linewidth=1)
        
        # 调整刻度字号
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # 保存单个性质的散点图
        output_path = os.path.join(scatter_dir, f"{prop}_scatter.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"散点图已保存到文件夹：{scatter_dir}(共{len(properties)}个性质,缩放因子: {scale_factor:.1f})")

def plot_bar_metrics(metrics, output_img, random_seed, figsize=(20, 8), property_list=None):
    """绘制训练集和测试集性能指标的簇状条形图, 包括R2, RMSE和Pearson相关系数"""
    
    # 只保留中文字体设置,移除全局字号设置
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    # ["SimSun", "DejaVu Sans"]
    plt.rcParams['axes.unicode_minus'] = False
    
    # 如果指定了property_list,只处理列表中的性质
    if property_list is not None:
        properties = [prop for prop in metrics.keys() if prop in property_list]
    else:
        properties = list(metrics.keys())
        
    if not properties:
        print("没有可用的性能指标数据")
        return
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 设置字体大小变量
    y_label_fontsize = 14
    title_fontsize = 16
    x_label_fontsize = 12
    legend_fontsize = 12
    tick_fontsize = 11
    
    # 设置颜色
    colors = {'训练集': '#2E86AB', '测试集': '#A23B72'}
    
    # R² 条形图
    x = np.arange(len(properties))
    width = 0.35
    
    min_test_r2 = 0
    for i, prop in enumerate(properties):
        # 修复：提取cn模式指标
        train_cn = metrics[prop].get('训练集', {}).get("cn", {})
        test_cn = metrics[prop].get('测试集', {}).get("cn", {})
        train_r2 = train_cn.get('R2', 0)
        test_r2 = test_cn.get('R2', 0)
        if test_r2 < min_test_r2:
            min_test_r2 = test_r2
        
        axes[0].bar(i - width/2, train_r2, width, label='训练集' if i == 0 else "", color=colors['训练集'])
        axes[0].bar(i + width/2, test_r2, width, label='测试集' if i == 0 else "", color=colors['测试集'])

    # axes[0].set_ylabel('R²', fontsize=y_label_fontsize)
    # axes[0].set_title('决定系数 (R²)', fontsize=title_fontsize, fontweight='bold', pad=15)
    # axes[0].set_xticks(x)
    # axes[0].set_xticklabels(properties, rotation=45, ha='right', fontsize=x_label_fontsize)
    # axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # axes[0].legend(fontsize=legend_fontsize)
    # axes[0].grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('R²', fontsize=y_label_fontsize)
    axes[0].set_title('决定系数 (R²)', fontsize=title_fontsize, fontweight='bold', pad=15)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(properties, rotation=45, ha='right', fontsize=x_label_fontsize)
    axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[0].legend(fontsize=legend_fontsize)
    axes[0].grid(True, alpha=0.3, axis='y')
    if min_test_r2 < -3:
        axes[0].set_ylim(-3, 1.5) 
    # y_data = []
    # for line in axes[0].get_lines():  # 遍历axes里所有绘制的线条
    #     y_data.extend(line.get_ydata())  # 收集所有y轴数据点

    # if y_data:  
    #     y_min = min(y_data)
    #     if y_min < -20:  
    #         axes[0].set_ylim(bottom=-2, top=1.2)  
    # else:
    #     # 无数据时的兜底(可选,防止坐标轴范围混乱)
        

    # MAPE 条形图
    for i, prop in enumerate(properties):
        # 修复：提取cn模式指标
        train_cn = metrics[prop].get('训练集', {}).get("cn", {})
        test_cn = metrics[prop].get('测试集', {}).get("cn", {})
        train_mape = train_cn.get('MAPE', 0)
        test_mape = test_cn.get('MAPE', 0)
        
        axes[1].bar(i - width/2, train_mape, width, label='训练集' if i == 0 else "", color=colors['训练集'])
        axes[1].bar(i + width/2, test_mape, width, label='测试集' if i == 0 else "", color=colors['测试集'])
    
    axes[1].set_ylabel('MAPE', fontsize=y_label_fontsize)
    axes[1].set_title('平均绝对百分比误差 (MAPE)', fontsize=title_fontsize, fontweight='bold', pad=15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(properties, rotation=45, ha='right', fontsize=x_label_fontsize)
    axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[1].legend(fontsize=legend_fontsize)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Pearson 相关系数条形图
    for i, prop in enumerate(properties):
        # 修复：提取cn模式指标
        train_cn = metrics[prop].get('训练集', {}).get("cn", {})
        test_cn = metrics[prop].get('测试集', {}).get("cn", {})
        train_pearson = train_cn.get('Pearson相关系数', 0)
        test_pearson = test_cn.get('Pearson相关系数', 0)
        
        # 处理NaN值
        train_pearson = 0 if np.isnan(train_pearson) else train_pearson
        test_pearson = 0 if np.isnan(test_pearson) else test_pearson
        
        axes[2].bar(i - width/2, train_pearson, width, label='训练集' if i == 0 else "", color=colors['训练集'])
        axes[2].bar(i + width/2, test_pearson, width, label='测试集' if i == 0 else "", color=colors['测试集'])
    
    axes[2].set_ylabel('Pearson相关系数', fontsize=y_label_fontsize)
    axes[2].set_title('Pearson相关系数', fontsize=title_fontsize, fontweight='bold', pad=15)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(properties, rotation=45, ha='right', fontsize=x_label_fontsize)
    axes[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[2].legend(fontsize=legend_fontsize)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"性能指标条形图已保存：{output_img}")
    
    # 生成测试结论Excel文件
    generate_test_conclusion_json(metrics, random_seed, output_img, property_list)

def generate_test_conclusion_json(metrics, random_seed, output_img, property_list=None):
    """生成测试结论JSON文件,包含每个性质的分表和按指标分类的整合表"""

    img_dir = os.path.dirname(output_img)
    json_path = os.path.join(img_dir, 'test_conclusion.json')

    try:
        properties = metrics.keys()
        if property_list is not None:
            properties = [prop for prop in metrics.keys() if prop in property_list]

        r2_summary = {}
        rmse_summary = {}
        mse_summary = {}
        mape_summary = {}
        pvalue_summary = {}
        pearson_summary = {}

        property_data = {}

        for prop_name in properties:
            test_metrics = metrics[prop_name].get('测试集', {}).get("cn", {})
            r2 = test_metrics.get('R2', 0)
            rmse = test_metrics.get('RMSE', 0)
            mse = test_metrics.get('MSE', 0)
            mape = test_metrics.get('MAPE', 0)
            p_value = test_metrics.get('P值', 1)
            pearson_corr = test_metrics.get('Pearson相关系数', 0)

            def _safe_val(v):
                if isinstance(v, (int, float, np.number)):
                    if np.isnan(v) or np.isinf(v):
                        return None
                    return float(v)
                return v

            r2_summary[prop_name] = _safe_val(r2)
            rmse_summary[prop_name] = _safe_val(rmse)
            mse_summary[prop_name] = _safe_val(mse)
            mape_summary[prop_name] = _safe_val(mape)
            pvalue_summary[prop_name] = _safe_val(p_value)
            pearson_summary[prop_name] = _safe_val(pearson_corr)

            property_data[prop_name] = {
                '测试集R2': _safe_val(r2),
                '测试集RMSE': _safe_val(rmse),
                '测试集MSE': _safe_val(mse),
                '测试集MAE': _safe_val(test_metrics.get('MAE', 0)),
                '测试集MAPE': _safe_val(mape),
                '测试集P值': _safe_val(p_value),
                '测试集Pearson相关系数': _safe_val(pearson_corr),
            }

        existing_data = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = {}

        existing_properties = existing_data.get('properties', {})
        for prop_name, data in property_data.items():
            if prop_name not in existing_properties:
                existing_properties[prop_name] = []
            existing_properties[prop_name].append(data)

        summary_data = {
            '随机种子': random_seed,
            'R2整合表': r2_summary,
            'RMSE整合表': rmse_summary,
            'MSE整合表': mse_summary,
            'MAPE整合表': mape_summary,
            'P值整合表': pvalue_summary,
            'Pearson整合表': pearson_summary,
        }

        existing_summaries = existing_data.get('summaries', [])
        existing_summaries.append(summary_data)

        output_data = {
            'properties': existing_properties,
            'summaries': existing_summaries,
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4, default=str)

        print(f"测试结论JSON文件已保存：{json_path}")
        print(f"包含以下整合表：R2整合表、RMSE整合表、MSE整合表、MAPE整合表、P值整合表、Pearson整合表")

    except Exception as e:
        print(f"生成测试结论JSON文件失败: {str(e)}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")

def clean_sheet_name(name):
    """清理sheet名称,确保符合Excel命名规范"""
    # 移除非法字符
    cleaned = re.sub(r'[\\/*?:[\]]', '', name)
    # 限制长度(Excel sheet名称最多31个字符)
    if len(cleaned) > 31:
        cleaned = cleaned[:31]
    # 如果名称为空,使用默认名称
    if not cleaned:
        cleaned = 'Sheet'
    return cleaned

def plot_clustered_bar_each_property(
    train_pred_file,    # 训练集预测结果Excel路径 (best_model_perform.xlsx)
    test_pred_file,     # 测试集预测结果Excel路径
    test_pred_sheet,    # 测试集预测结果sheet名
    test_actual_file,   # 测试集实际值Excel路径
    test_actual_sheet,  # 测试集实际值sheet名
    output_img_prefix,  # 输出图片路径前缀
    property_list=None, # 指定要处理的性质列表
    max_samples_per_plot=100,  # 每个图最多显示的样品数
    figsize=(18, 10),    # 增大的图片大小
    train_data_dict: dict = None,  # 训练集数据字典 {prop: DataFrame}
    test_data_dict = None,  # 测试集预测DataFrame,替代test_pred_file
    test_actual_df = None,  # 测试集实际值DataFrame,替代test_actual_file
    test_ids_per_prop: dict = None,  # 逐性质测试集id {prop: [id,...]},用于按id对齐实际/预测
):
    """为每个性质生成簇状条形图,展示每个样品的实际值和预测值"""
    
    # 读取训练集预测数据
    train_data = {}
    if train_data_dict is not None:
        for prop, val in train_data_dict.items():
            if property_list is not None and prop not in property_list:
                continue
            if isinstance(val, dict) and 'predictions' in val:
                df = pd.DataFrame(val['predictions'])
            elif isinstance(val, pd.DataFrame):
                df = val
            else:
                df = pd.DataFrame(val)
            exp_cols = [col for col in df.columns if col.endswith('_exp')]
            pred_cols = [col for col in df.columns if col.endswith('_pred')]
            if exp_cols and pred_cols:
                train_data[prop] = {
                    'actual': df[exp_cols[0]].values,
                    'pred': df[pred_cols[0]].values,
                    'source': '训练集'
                }
    else:
        print(f"警告: 未提供训练集数据字典 train_data_dict")
    
    # 读取测试集预测数据
    if test_data_dict is not None:
        if isinstance(test_data_dict, pd.DataFrame):
            test_pred_df = test_data_dict
        else:
            test_pred_df = pd.DataFrame(test_data_dict)
    else:
        raise ValueError("未提供测试集预测数据 test_data_dict")
    # 预测列 = 排除元数据列后的所有列（预测结果表元数据列为：样本编号/数据集类型/原始编号/样本标识）
    _meta_cols = ('样本编号', '数据集类型', '原始编号', '样本标识')
    test_pred_cols = [col for col in test_pred_df.columns if col not in _meta_cols]
    
    if property_list is not None:
        test_pred_cols = [col for col in test_pred_cols if col in property_list]
    
    if not test_pred_cols:
        raise ValueError("在测试集预测数据中没有找到指定的性质")
        
    test_pred_data = test_pred_df[test_pred_cols].values
    
    if test_actual_df is not None:
        _test_actual_df_2 = test_actual_df
    else:
        raise ValueError("未提供测试集实际数据 test_actual_df")
    test_actual_candidate_cols = _test_actual_df_2.iloc[:, 1:].columns.tolist()
    test_actual_matched_cols = [col for col in test_pred_cols if col in test_actual_candidate_cols]
    
    if not test_actual_matched_cols:
        raise ValueError("测试集实际数据表中没有找到与预测数据表匹配的字段")
    
    test_actual_data = _test_actual_df_2[test_actual_matched_cols].values
    
    # 测试集 id 列识别（用于按性质测试集划分逐性质对齐）
    def _first_id_col(df):
        if df is None or len(df) == 0:
            return None
        for col in ['样本编号', '样本标识', '原始编号']:
            if col in df.columns:
                return col
        return None
    pred_id_col = _first_id_col(test_pred_df)
    actual_id_col = _first_id_col(_test_actual_df_2)

    # 准备测试集数据（按性质测试集id逐性质对齐，无匹配样本或长度不一致的性质跳过）
    for i, prop in enumerate(test_actual_matched_cols):
        y_actual = test_actual_data[:, i]
        y_pred = test_pred_data[:, i]

        if test_ids_per_prop and prop in test_ids_per_prop:
            prop_test_ids = set(str(x) for x in test_ids_per_prop[prop])
            if pred_id_col and actual_id_col:
                pred_ids = test_pred_df[pred_id_col].astype(str).values
                actual_ids = _test_actual_df_2[actual_id_col].astype(str).values
                actual_by_id = {aid: av for aid, av in zip(actual_ids, y_actual)}
                y_pred_f, y_actual_f = [], []
                for pid, pv in zip(pred_ids, y_pred):
                    if pid in prop_test_ids and pid in actual_by_id:
                        y_pred_f.append(pv)
                        y_actual_f.append(actual_by_id[pid])
                if not y_pred_f:
                    print(f"跳过性质 '{prop}' 的簇状条形图：测试集无匹配样本")
                    continue
                y_actual = np.array(y_actual_f)
                y_pred = np.array(y_pred_f)
            else:
                # 无 id 列可用：长度不一致时跳过，避免广播错误
                if len(y_actual) != len(y_pred):
                    print(f"跳过性质 '{prop}' 的簇状条形图：测试集实际值 {len(y_actual)} 行与预测值 {len(y_pred)} 行不一致")
                    continue
        elif len(y_actual) != len(y_pred):
            print(f"跳过性质 '{prop}' 的簇状条形图：测试集实际值 {len(y_actual)} 行与预测值 {len(y_pred)} 行不一致")
            continue

        test_data = {
            'actual': y_actual,
            'pred': y_pred,
            'source': '测试集'
        }
        # 如果训练集中已有该性质,合并数据
        if prop in train_data:
            train_test_data = {
                '训练集': train_data[prop],
                '测试集': test_data
            }
            train_data[prop] = train_test_data
        else:
            train_data[prop] = {'测试集': test_data}
    
    # 计算R2指标用于图例显示
    metrics = {}
    for prop, data_dict in train_data.items():
        # 如果指定了property_list,只处理列表中的性质
        if property_list is not None and prop not in property_list:
            continue
            
        metrics[prop] = {}
        for source in ['训练集', '测试集']:
            if source in data_dict:
                source_data = data_dict[source]
                y_true = source_data['actual']
                y_pred = source_data['pred']
                if len(y_true) != len(y_pred):
                    print(f"跳过性质 '{prop}' 的{source}指标计算：实际值 {len(y_true)} 行与预测值 {len(y_pred)} 行不一致")
                    continue
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                y_true_clean = y_true[mask]
                y_pred_clean = y_pred[mask]
                
                if len(y_true_clean) > 0:
                    r2 = r2_score(y_true_clean, y_pred_clean)
                    metrics[prop][source] = {'R2': r2}
    
    # 中文字体设置和增大全局字号
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    # ["SimSun","DejaVu Sans"]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 14
    })
    
    # 创建簇状条形图文件夹
    bar_dir = f"{output_img_prefix}_bar"
    os.makedirs(bar_dir, exist_ok=True)
    
    # 为每个性质生成簇状条形图
    for prop, data_dict in train_data.items():
        # 如果指定了property_list,只处理列表中的性质
        if property_list is not None and prop not in property_list:
            continue
            
        print(f"生成性质 '{prop}' 的簇状条形图...")

        # 清理性质名称中的特殊字符
        clean_prop = prop
        if clean_prop != prop:
            print(f"  注意: 性质名称 '{prop}' 包含特殊字符,已清理为 '{clean_prop}'")
        
        prop = clean_prop

        # 准备数据
        all_samples = []
        all_values = []
        all_types = []  # '实际值' 或 '预测值'
        all_sources = []  # '训练集' 或 '测试集'
        sample_indices = []  # 样品编号

        n_train = 0  # 训练集样本数（训练集被跳过时为0）

        # 处理训练集数据
        if '训练集' in data_dict:
            train_actual = data_dict['训练集']['actual']
            train_pred = data_dict['训练集']['pred']
            if len(train_actual) != len(train_pred):
                print(f"跳过性质 '{prop}' 的训练集绘图：实际值 {len(train_actual)} 行与预测值 {len(train_pred)} 行不一致")
            else:
                # 过滤NaN值
                mask = ~(np.isnan(train_actual) | np.isnan(train_pred))
                train_actual_clean = train_actual[mask]
                train_pred_clean = train_pred[mask]

                n_train = len(train_actual_clean)

                # 添加训练集实际值
                for i in range(n_train):
                    all_samples.append(f"Train_{i+1}")
                    all_values.append(train_actual_clean[i])
                    all_types.append('实际值')
                    all_sources.append('训练集')
                    sample_indices.append(i)

                # 添加训练集预测值
                for i in range(n_train):
                    all_samples.append(f"Train_{i+1}")
                    all_values.append(train_pred_clean[i])
                    all_types.append('预测值')
                    all_sources.append('训练集')
                    sample_indices.append(i)
        
        # 处理测试集数据
        if '测试集' in data_dict:
            test_actual = data_dict['测试集']['actual']
            test_pred = data_dict['测试集']['pred']
            if len(test_actual) != len(test_pred):
                print(f"跳过性质 '{prop}' 的测试集绘图：实际值 {len(test_actual)} 行与预测值 {len(test_pred)} 行不一致")
            else:
                # 过滤NaN值
                mask = ~(np.isnan(test_actual) | np.isnan(test_pred))
                test_actual_clean = test_actual[mask]
                test_pred_clean = test_pred[mask]

                n_test = len(test_actual_clean)

                # 添加测试集实际值
                for i in range(n_test):
                    all_samples.append(f"Test_{i+1}")
                    all_values.append(test_actual_clean[i])
                    all_types.append('实际值')
                    all_sources.append('测试集')
                    sample_indices.append(n_train + i)

                # 添加测试集预测值
                for i in range(n_test):
                    all_samples.append(f"Test_{i+1}")
                    all_values.append(test_pred_clean[i])
                    all_types.append('预测值')
                    all_sources.append('测试集')
                    sample_indices.append(n_train + i)
        
        # 如果样品太多,分批绘制
        total_samples = len(set(all_samples))
        if total_samples == 0:
            print(f"跳过性质 '{prop}' 的簇状条形图：训练/测试集均无有效样本")
            continue

        if total_samples > max_samples_per_plot:
            print(f"警告: 性质 '{prop}' 有 {total_samples} 个样品,超过最大限制 {max_samples_per_plot},将分批绘制")
            
            # 分批处理
            unique_samples = sorted(set(all_samples), key=lambda x: (0 if x.startswith('Train') else 1, int(x.split('_')[1])))
            
            for batch_num, batch_start in enumerate(range(0, total_samples, max_samples_per_plot)):
                batch_end = min(batch_start + max_samples_per_plot, total_samples)
                batch_samples = unique_samples[batch_start:batch_end]
                
                # 筛选当前批次的数据
                batch_mask = [sample in batch_samples for sample in all_samples]
                batch_all_samples = [all_samples[i] for i in range(len(all_samples)) if batch_mask[i]]
                batch_all_values = [all_values[i] for i in range(len(all_values)) if batch_mask[i]]
                batch_all_types = [all_types[i] for i in range(len(all_types)) if batch_mask[i]]
                batch_all_sources = [all_sources[i] for i in range(len(all_sources)) if batch_mask[i]]
                batch_sample_indices = [sample_indices[i] for i in range(len(sample_indices)) if batch_mask[i]]
                
                # 绘制当前批次
                _plot_single_property_clustered_bar(
                    prop, batch_all_samples, batch_all_values, batch_all_types, 
                    batch_all_sources, batch_sample_indices, metrics,
                    f"{bar_dir}/{prop}_part{batch_num+1}_bar.png", figsize
                )
        else:
            # 直接绘制所有样品
            _plot_single_property_clustered_bar(
                prop, all_samples, all_values, all_types, all_sources, sample_indices, metrics,
                f"{bar_dir}/{prop}_bar.png", figsize
            )

def _plot_single_property_clustered_bar(
    prop, all_samples, all_values, all_types, all_sources, sample_indices, metrics,
    output_img, figsize=(18, 10)
):
    """为单个性质绘制簇状条形图,图例显示训练测试集的R2"""
    
    # 创建DataFrame以便于绘图
    df = pd.DataFrame({
        '样品': all_samples,
        '数值': all_values,
        '类型': all_types,
        '数据集': all_sources,
        '样品索引': sample_indices
    })
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置颜色
    colors = {
        ('训练集', '实际值'): '#1f77b4',  # 深蓝色
        ('训练集', '预测值'): '#aec7e8',  # 浅蓝色
        ('测试集', '实际值'): '#d62728',  # 深红色
        ('测试集', '预测值'): '#ff9896'   # 浅红色
    }
    
    # 获取唯一的样品并按索引排序
    unique_samples = sorted(set(all_samples), key=lambda x: (0 if x.startswith('Train') else 1, int(x.split('_')[1])))
    
    # 设置条形图的位置
    x_pos = np.arange(len(unique_samples))
    bar_width = 0.35
    
    # 为每个样品绘制实际值和预测值的条形
    for i, sample in enumerate(unique_samples):
        sample_data = df[df['样品'] == sample]
        
        # 获取实际值和预测值
        actual_val = sample_data[sample_data['类型'] == '实际值']['数值'].values
        pred_val = sample_data[sample_data['类型'] == '预测值']['数值'].values
        
        if len(actual_val) > 0 and len(pred_val) > 0:
            dataset = sample_data['数据集'].iloc[0]  # 训练集或测试集
            
            # 绘制实际值条形
            ax.bar(i - bar_width/2, actual_val[0], bar_width, 
                   color=colors[(dataset, '实际值')], label='实际值' if i == 0 else "")
            
            # 绘制预测值条形
            ax.bar(i + bar_width/2, pred_val[0], bar_width, 
                   color=colors[(dataset, '预测值')], label='预测值' if i == 0 else "")
    
    # 添加训练集和测试集的分隔线
    train_samples = [s for s in unique_samples if s.startswith('Train')]
    test_samples = [s for s in unique_samples if s.startswith('Test')]
    
    if train_samples and test_samples:
        separator_pos = len(train_samples) - 0.5
        ax.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        
        # 添加数据集标签
        train_center = len(train_samples) / 2 - 0.5
        test_center = len(train_samples) + len(test_samples) / 2 - 0.5
        
        ax.text(train_center, ax.get_ylim()[1] * 0.95, '训练集', 
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(test_center, ax.get_ylim()[1] * 0.95, '测试集', 
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # 获取R2值用于图例
    train_r2 = metrics.get(prop, {}).get('训练集', {}).get('R2', 0)
    test_r2 = metrics.get(prop, {}).get('测试集', {}).get('R2', 0)
    
    # 设置图形属性
    ax.set_xlabel('样品', fontsize=16)
    ax.set_ylabel('数值', fontsize=16)
    ax.set_title(f'{prop} - 预测值与实际值对比', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(unique_samples, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=14)
    
    # 设置图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[('训练集', '实际值')], label=f'训练集-实际值 (R²={train_r2:.3f})'),
        Patch(facecolor=colors[('训练集', '预测值')], label='训练集-预测值'),
        Patch(facecolor=colors[('测试集', '实际值')], label=f'测试集-实际值 (R²={test_r2:.3f})'),
        Patch(facecolor=colors[('测试集', '预测值')], label='测试集-预测值')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=16)
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='y', linewidth=1)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(output_img, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"簇状条形图已保存：{output_img}")

def plot_pred_actual(
    pred_file,        # 预测结果Excel路径
    pred_sheet,       # 预测结果sheet名
    actual_file,      # 实际值Excel路径
    actual_sheet,     # 实际值sheet名
    output_img,       # 输出图片路径
    property_list=None, # 指定要处理的性质列表
    figsize=(18, 12), # 增大的图片大小
    cols_per_row=2,    # 每行子图数
    pred_df=None,      # 预测结果DataFrame,替代pred_file
    actual_df=None,    # 实际值DataFrame,替代actual_file
):
    """生成多个性质的预测-实际对比散点图(含趋势线和评估指标)"""
    if pred_df is not None:
        _pred_df = pred_df
    else:
        raise ValueError("未提供预测数据 pred_df")
    pred_cols = _pred_df.iloc[:, 3:].columns.tolist()
    
    # 如果指定了property_list,筛选预测数据列
    if property_list is not None:
        pred_cols = [col for col in pred_cols if col in property_list]
    
    if not pred_cols:
        raise ValueError("在预测数据中没有找到指定的性质")
        
    pred_data = _pred_df[pred_cols].values
    
    if actual_df is not None:
        _actual_df = actual_df
    else:
        raise ValueError("未提供实际数据 actual_df")
    actual_candidate_cols = _actual_df.iloc[:, 1:].columns.tolist()
    actual_matched_cols = [col for col in pred_cols if col in actual_candidate_cols]

    if not actual_matched_cols:
        raise ValueError("实际数据表中没有找到与预测数据表匹配的字段")

    actual_data = _actual_df[actual_matched_cols].values

    if len(pred_data) != len(actual_data):
        raise ValueError(f"样本数不匹配：预测{len(pred_data)} vs 实际{len(actual_data)}")

    # 子图布局
    n = len(pred_cols)
    rows = (n + cols_per_row - 1) // cols_per_row
    cols = cols_per_row

    # 中文字体设置和增大全局字号
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    # ["SimSun","DejaVu Sans"]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    # 创建画布
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # 绘制每个性质的对比图
    for i, prop in enumerate(pred_cols):
        ax = axes[i]
        y_pred = pred_data[:, i]
        y_true = actual_data[:, i]
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        yp, yt = y_pred[mask], y_true[mask]

        if len(yp) == 0:
            ax.text(0.5, 0.5, f"{prop}\n无有效数据", ha='center', va='center', 
                    transform=ax.transAxes, fontsize=16)
            ax.set_title(prop, fontsize=20)
            continue

        # 计算指标
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        mse = mean_squared_error(yt, yp)
        rmse = np.sqrt(mse)
        mape = calculate_mape(yt, yp)

        # 绘图
        ax.scatter(yt, yp, alpha=0.7, s=80, color='#2E86AB')
        min_val, max_val = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='理想线 (x=y)')

        # 添加指标文本
        ax.text(0.05, 0.95, f'R²={r2:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}\nMAPE={mape:.2f}%',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                verticalalignment='top', fontsize=12, linespacing=1.5)

        # 标题和标签
        ax.set_title(prop, fontsize=28, fontweight='bold')
        ax.set_xlabel('实际值', fontsize=14)
        ax.set_ylabel('预测值', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    # 删除未使用的子图
    for i in range(n, rows * cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_img, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"对比图已保存：{output_img}(共{len(pred_cols)}个性质)")

# 示例调用
if __name__ == "__main__":
    print("可视化模块已更新为DataFrame直传模式,请通过Excute_pipe.py调用")
