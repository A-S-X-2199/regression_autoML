"""
SVD降维模块
包含标准SVD和截断SVD两种降维方法
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import pickle
import os
import json
from .feature_schema import split_feature_columns


class StandardSVDWrapper:
    """标准SVD包装器，基于np.linalg.svd精确分解，提供与TruncatedSVD兼容的接口"""

    def __init__(self, Vt: np.ndarray, s: np.ndarray, n_components: int, n_samples: int):
        self.components_ = Vt[:n_components]
        self.singular_values_ = s[:n_components]
        self.n_components = n_components
        self._Vt_full = Vt
        self._s_full = s
        self._n_samples = n_samples
        self.algorithm = 'exact'

        explained_variance = (s[:n_components] ** 2) / (n_samples - 1)
        self.explained_variance_ = explained_variance
        total_variance = np.sum(s ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = explained_variance / total_variance

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.components_.T

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        return X_transformed @ self.components_


class SVDReducer:
    """SVD降维器（包含标准SVD和截断SVD）"""
    
    def __init__(self, n_features_for_reduction: int = 50, svd_type: str = 'svd'):
        """
        初始化SVD降维器
        
        Args:
            n_features_for_reduction: 参与降维的特征数量
            svd_type: 'svd'（标准SVD）或 'tsvd'（截断SVD）
        """
        self.n_features_for_reduction = n_features_for_reduction
        self.svd_type = svd_type.lower()
        self.reduction_type = 'svd' if self.svd_type == 'svd' else 'tsvd'
        
        if self.svd_type not in ['svd', 'tsvd']:
            raise ValueError(f"svd_type必须是'svd'或'tsvd'，当前为: {svd_type}")
        
    def get_summary_filename(self, output_dir: str) -> str:
        """获取汇总文件名"""
        if self.svd_type == 'svd':
            return f"{output_dir}/svd_summary.json"
        else:  # tsvd
            return f"{output_dir}/tsvd_summary.json"
    
    def split_features(self, X: pd.DataFrame) -> tuple:
        """按输入中的显式标记拆分前部分与参与降维的后部分。"""
        return split_feature_columns(X.columns)
    
    def reduce_fold(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                   y_train=None, ratio: float = 0.5) -> tuple:
        """
        对交叉验证的一折数据应用SVD降维
        返回: (X_train_processed, X_val_processed, svd_info)
        """
        # 分离特征
        front_features, back_features = self.split_features(X_train)
        front_feature_count = len(front_features)
        
        # 前部分特征
        front_train = X_train[front_features] if front_feature_count > 0 else pd.DataFrame()
        front_val = X_val[front_features] if front_feature_count > 0 else pd.DataFrame()
        
        # 后部分特征
        back_train = X_train[back_features].values.astype(np.float64) if back_features else np.array([]).reshape(len(X_train), 0)
        back_val = X_val[back_features].values.astype(np.float64) if back_features else np.array([]).reshape(len(X_val), 0)
        
        if back_train.shape[1] == 0:
            # 没有后部分特征，直接返回前部分特征
            return front_train, front_val, None
        
        # 标准化
        # scaler = StandardScaler()
        # back_train_scaled = scaler.fit_transform(back_train)
        # back_val_scaled = scaler.transform(back_val)

        back_train_scaled = back_train
        back_val_scaled = back_val
        
        # 计算实际维度
        max_dimension = min(back_train_scaled.shape[1], back_train_scaled.shape[0])
        actual_dimension = max(1, min(max_dimension, round(ratio * back_train_scaled.shape[1])))
        
        # SVD降维
        n_components = actual_dimension

        if self.svd_type == 'svd':
            n_components = min(n_components, min(back_train_scaled.shape) - 1)
            if n_components <= 0:
                n_components = 1

            U, s, Vt = np.linalg.svd(back_train_scaled, full_matrices=False)
            svd_scores_train = U[:, :n_components] * s[:n_components]
            svd_scores_val = back_val_scaled @ Vt[:n_components].T
            svd = StandardSVDWrapper(Vt, s, n_components, back_train_scaled.shape[0])
        else:
            svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
            svd_scores_train = svd.fit_transform(back_train_scaled)
            svd_scores_val = svd.transform(back_val_scaled)
        
        # 创建SVD特征
        svd_columns = [f'SVD_主成分{i+1}' for i in range(n_components)]
        svd_train_df = pd.DataFrame(svd_scores_train, columns=svd_columns, index=X_train.index)
        svd_val_df = pd.DataFrame(svd_scores_val, columns=svd_columns, index=X_val.index)
        
        # 合并特征
        if front_feature_count > 0:
            front_train = front_train.reset_index(drop=True)
            front_val = front_val.reset_index(drop=True)
            svd_train_df = svd_train_df.reset_index(drop=True)
            svd_val_df = svd_val_df.reset_index(drop=True)
            
            final_train = pd.concat([front_train, svd_train_df], axis=1)
            final_val = pd.concat([front_val, svd_val_df], axis=1)
        else:
            final_train = svd_train_df
            final_val = svd_val_df
        
        # 保存降维信息
        svd_info = {
            # 'scaler': scaler,
            'svd': svd,
            'svd_type': self.svd_type,
            'n_components': n_components,
            'svd_columns': svd_columns,
            'front_columns': front_train.columns.tolist() if front_feature_count > 0 else [],
            'reduction_ratio': ratio,
            'actual_dimension': actual_dimension,
            'explained_variance': svd.explained_variance_,
            'explained_variance_ratio': svd.explained_variance_ratio_,
            'singular_values': svd.singular_values_,
            'components': svd.components_
        }
        
        return final_train, final_val, svd_info
    
    def reduce_full_data(self, output_dir, training_params_dir, X: pd.DataFrame, y=None, ratio: float = 0.5, 
                        property_name: str = None) -> tuple:
        """对完整数据集应用SVD降维"""
        # 分离特征
        front_features, back_features = self.split_features(X)
        front_feature_count = len(front_features)
        
        # 动态设置汇总文件路径
        self.svd_summary_file = self.get_summary_filename(output_dir)
            
        # 前部分特征
        front_data = X[front_features] if front_feature_count > 0 else pd.DataFrame()
        
        # 后部分特征
        back_data = X[back_features].values.astype(np.float64) if back_features else np.array([]).reshape(len(X), 0)
        
        if back_data.shape[1] == 0:
            # 没有后部分特征
            return front_data, None
        
        # 标准化
        # scaler = StandardScaler()
        # back_data_scaled = scaler.fit_transform(back_data)
        back_data_scaled = back_data

        
        # 计算实际维度
        max_dimension = min(back_data_scaled.shape[1], back_data_scaled.shape[0])
        actual_dimension = max(1, min(max_dimension, round(ratio * back_data_scaled.shape[1])))
        
        # SVD降维
        n_components = actual_dimension

        if self.svd_type == 'svd':
            n_components = min(n_components, min(back_data_scaled.shape) - 1)
            if n_components <= 0:
                n_components = 1

            U, s, Vt = np.linalg.svd(back_data_scaled, full_matrices=False)
            svd_scores = U[:, :n_components] * s[:n_components]
            svd = StandardSVDWrapper(Vt, s, n_components, back_data_scaled.shape[0])
        else:
            svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
            svd_scores = svd.fit_transform(back_data_scaled)
        
        # 创建SVD特征
        svd_columns = [f'SVD_主成分{i+1}' for i in range(n_components)]
        svd_df = pd.DataFrame(svd_scores, columns=svd_columns, index=X.index)
        
        # 合并特征
        if front_feature_count > 0:
            front_data = front_data.reset_index(drop=True)
            svd_df = svd_df.reset_index(drop=True)
            final_data = pd.concat([front_data, svd_df], axis=1)
        else:
            final_data = svd_df
        
        # 计算主成分与y的相关系数
        svd_correlations = {}
        if y is not None:
            for i, col in enumerate(svd_columns):
                correlation = np.corrcoef(svd_scores[:, i], y)[0, 1]
                svd_correlations[col] = correlation
        
        # 计算累计方差解释率
        explained_variance_ratio = svd.explained_variance_ratio_
        cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # 计算重构误差
        X_reconstructed = svd.inverse_transform(svd_scores)
        reconstruction_error = np.mean(np.abs(back_data_scaled - X_reconstructed))
        
        # 保存降维信息
        svd_info = {
            # 'scaler': scaler,
            'svd': svd,
            'svd_type': self.svd_type,
            'n_components': n_components,
            'svd_columns': svd_columns,
            'front_columns': front_data.columns.tolist() if front_feature_count > 0 else [],
            'back_columns': back_features,
            'reduction_ratio': ratio,
            'actual_dimension': actual_dimension,
            'back_feature_count': len(back_features),
            'svd_correlations': svd_correlations,
            'explained_variance': svd.explained_variance_.tolist(),
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'singular_values': svd.singular_values_.tolist(),
            'components': svd.components_,
            'cumulative_explained_variance_ratio': cumulative_explained_variance_ratio.tolist(),
            'reconstruction_error': reconstruction_error,
            'total_variance_explained': np.sum(explained_variance_ratio),
            'algorithm': svd.algorithm
        }
        
        # 保存到文件
        if property_name:
            self._save_svd_info(training_params_dir, svd_info, property_name)
            # 生成SVD详细报告
            self._save_svd_detailed_report(svd_info, property_name, X, y)
        
        return final_data, svd_info
    
    def _save_svd_info(self, training_params_dir, svd_info: dict, property_name: str):
        """保存SVD信息到文件"""
        try:
            training_params_dir = training_params_dir
            os.makedirs(training_params_dir, exist_ok=True) 

            # 安全处理属性名称
            safe_prop = property_name
            replace_map = {"（": "_", "）": "_", "/": "_"}
            for old, new in replace_map.items():
                safe_prop = safe_prop.replace(old, new)

            filename = f"{training_params_dir}/{self.reduction_type}_params_{safe_prop}.pkl"
            
            with open(filename, 'wb') as f:
                pickle.dump(svd_info, f)
            
            print(f"{self.reduction_type.upper()}参数已保存到 {filename}")
        except Exception as e:
            print(f"保存{self.reduction_type.upper()}参数失败: {e}")
    
    def _save_svd_detailed_report(self, svd_info: dict, property_name: str, X: pd.DataFrame, y=None):
        """保存SVD详细报告到JSON文件"""
        try:
            summary_file = self.svd_summary_file

            prop_data = self._build_svd_property_data(svd_info, property_name, X, y)

            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
                if isinstance(all_data, list):
                    all_data = {}
            else:
                all_data = {}

            all_data[property_name] = prop_data

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)

            print(f"{self.reduction_type.upper()}详细报告已保存到 {summary_file} 的 [{property_name}] 条目中")

        except Exception as e:
            print(f"保存{self.reduction_type.upper()}详细报告到JSON时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _build_svd_property_data(self, svd_info: dict, property_name: str, X: pd.DataFrame, y=None):
        """构建单个性质的SVD数据字典"""
        prop_data = {}

        max_singular = svd_info['singular_values'][0] if svd_info['singular_values'] else 0
        min_singular = svd_info['singular_values'][-1] if svd_info['singular_values'] else 0
        condition_number = max_singular / min_singular if min_singular > 0 else float('inf')
        if isinstance(condition_number, float) and np.isinf(condition_number):
            condition_number = None

        basic_info = {
            '属性名称': property_name,
            '降维类型': f'{self.reduction_type.upper()} ({svd_info["svd_type"].upper()})',
            '原始后部特征数量': svd_info['back_feature_count'],
            'SVD降维比例': svd_info['reduction_ratio'],
            '实际SVD维度': svd_info['actual_dimension'],
            '算法': svd_info['algorithm'],
            '总方差解释率': float(svd_info['total_variance_explained']) * 100,
            '重构误差': float(svd_info['reconstruction_error']),
            '最大奇异值': float(max_singular),
            '最小奇异值': float(min_singular) if min_singular > 0 else None
        }
        prop_data['basic_info'] = basic_info

        variance_data = []
        for i, (singular_value, var_ratio, cum_ratio) in enumerate(zip(
            svd_info['singular_values'],
            svd_info['explained_variance_ratio'],
            svd_info['cumulative_explained_variance_ratio']
        )):
            svd_col = f'SVD_主成分{i+1}'
            correlation = svd_info['svd_correlations'].get(svd_col) if svd_info.get('svd_correlations') else None
            if isinstance(correlation, float) and np.isnan(correlation):
                correlation = None

            ev = svd_info['explained_variance'][i] if i < len(svd_info['explained_variance']) else None
            if isinstance(ev, float) and np.isnan(ev):
                ev = None

            cond_ratio = singular_value / svd_info['singular_values'][0] if svd_info['singular_values'] and i > 0 else 1.0

            variance_data.append({
                '成分': f'SVD{i+1}',
                '奇异值': float(singular_value),
                '方差解释率': float(var_ratio) * 100,
                '累计方差解释率': float(cum_ratio) * 100,
                '方差贡献': ev,
                '与y的相关系数': correlation,
                '条件数比例': float(cond_ratio)
            })
        prop_data['variance'] = variance_data

        decay_data = []
        if len(svd_info['singular_values']) > 1:
            total_singular_energy = np.sum(np.square(svd_info['singular_values']))

            for k in range(1, min(11, len(svd_info['singular_values']))):
                partial_energy = np.sum(np.square(svd_info['singular_values'][:k]))
                energy_ratio = partial_energy / total_singular_energy

                decay_data.append({
                    '保留成分数': k,
                    '奇异值能量比': float(energy_ratio) * 100,
                    '剩余能量比': (1.0 - float(energy_ratio)) * 100,
                    '累计方差解释率': float(svd_info['cumulative_explained_variance_ratio'][k-1]) * 100
                })
        prop_data['decay'] = decay_data

        loadings_data = []
        if 'components' in svd_info and 'back_columns' in svd_info:
            components = svd_info['components']
            back_columns = svd_info['back_columns']

            if components is not None and len(components) > 0 and back_columns:
                n_components = components.shape[0]
                n_top_features = min(10, len(back_columns))

                for svd_idx in range(n_components):
                    svd_loadings = components[svd_idx]
                    abs_loadings = np.abs(svd_loadings)
                    top_indices = np.argsort(abs_loadings)[-n_top_features:][::-1]
                    max_abs = float(np.max(abs_loadings)) if np.max(abs_loadings) > 0 else 1.0

                    for rank, idx in enumerate(top_indices):
                        lv = float(svd_loadings[idx])
                        alv = float(abs_loadings[idx])
                        loadings_data.append({
                            'SVD成分': f'SVD{svd_idx+1}',
                            '排名': rank + 1,
                            '特征名称': str(back_columns[idx]),
                            '载荷值': lv,
                            '载荷绝对值': alv,
                            '相对重要性': alv / max_abs
                        })
        prop_data['loadings'] = loadings_data

        condition_data = None
        if len(svd_info['singular_values']) > 1:
            condition_data = {
                '最大奇异值': float(max_singular),
                '最小奇异值': float(min_singular) if min_singular > 0 else None,
                '条件数': condition_number,
                '数值稳定性评估': '良好' if (condition_number is not None and condition_number < 1000) else ('中等' if (condition_number is not None and condition_number < 1e6) else '较差')
            }
        prop_data['condition_data'] = condition_data

        return prop_data
    
    def get_svd_summary_statistics(self, svd_info: dict) -> dict:
        """获取SVD汇总统计信息"""
        summary = {
            'property_name': svd_info.get('property_name', 'Unknown'),
            'svd_type': svd_info['svd_type'],
            'n_components': svd_info['n_components'],
            'original_features': svd_info['back_feature_count'],
            'reduction_ratio': svd_info['reduction_ratio'],
            'total_variance_explained': svd_info['total_variance_explained'] * 100,
            'first_component_variance': svd_info['explained_variance_ratio'][0] * 100 if svd_info['explained_variance_ratio'] else 0,
            'reconstruction_error': svd_info.get('reconstruction_error', 0),
            'condition_number': svd_info['singular_values'][0] / svd_info['singular_values'][-1] if len(svd_info['singular_values']) > 1 and svd_info['singular_values'][-1] > 0 else float('inf'),
            'has_correlations': bool(svd_info.get('svd_correlations'))
        }
        
        # 如果有相关系数，添加相关系数信息
        if svd_info.get('svd_correlations'):
            correlations = list(svd_info['svd_correlations'].values())
            summary['max_correlation'] = max(correlations, key=abs)
            summary['avg_abs_correlation'] = np.mean([abs(c) for c in correlations])
        
        return summary
    
    def create_visualization_data(self, svd_info: dict, X: pd.DataFrame, y=None) -> dict:
        """创建可视化数据"""
        viz_data = {}
        
        # 1. 碎石图数据（奇异值衰减）
        viz_data['scree_plot'] = {
            'components': [f'SVD{i+1}' for i in range(svd_info['n_components'])],
            'singular_values': svd_info['singular_values'],
            'explained_variance_ratio': svd_info['explained_variance_ratio'],
            'cumulative_explained_variance_ratio': svd_info['cumulative_explained_variance_ratio']
        }
        
        # 2. 能量累积曲线
        if len(svd_info['singular_values']) > 1:
            energies = []
            total_energy = np.sum(np.square(svd_info['singular_values']))
            
            for k in range(1, len(svd_info['singular_values']) + 1):
                partial_energy = np.sum(np.square(svd_info['singular_values'][:k]))
                energies.append(partial_energy / total_energy)
            
            viz_data['energy_plot'] = {
                'num_components': list(range(1, len(svd_info['singular_values']) + 1)),
                'energy_ratios': energies
            }
        
        # 3. 载荷图数据（前两个SVD成分）
        if svd_info['n_components'] >= 2:
            viz_data['loadings_plot'] = {
                'features': svd_info['back_columns'],
                'svd1_loadings': svd_info['components'][0, :].tolist() if len(svd_info['components']) > 0 else [],
                'svd2_loadings': svd_info['components'][1, :].tolist() if len(svd_info['components']) > 1 else []
            }
        
        # 4. SVD得分数据
        if X is not None and svd_info.get('back_columns'):
            back_data = X[svd_info['back_columns']].values.astype(np.float64)
            # scaler = StandardScaler()
            # back_data_scaled = scaler.fit_transform(back_data)
            back_data_scaled = back_data
            svd_scores = svd_info['svd'].transform(back_data_scaled)
            
            viz_data['scores_plot'] = {
                'samples': list(range(len(svd_scores))),
                'svd1_scores': svd_scores[:, 0].tolist() if svd_info['n_components'] >= 1 else [],
                'svd2_scores': svd_scores[:, 1].tolist() if svd_info['n_components'] >= 2 else []
            }
            
            if y is not None:
                viz_data['scores_plot']['y_values'] = y.tolist()
        
        return viz_data
