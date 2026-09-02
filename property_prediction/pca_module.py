import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import os
import json
from .feature_schema import split_feature_columns

class PCAReducer:
    """PCA降维器"""
    
    def __init__(self, n_features_for_reduction: int = 50):
        self.n_features_for_reduction = n_features_for_reduction
        self.reduction_type = 'pca'
    
    def split_features(self, X: pd.DataFrame) -> tuple:
        """按输入中的显式标记拆分前部分与参与降维的后部分。"""
        return split_feature_columns(X.columns)
    
    def reduce_fold(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                   y_train=None, ratio: float = 0.5) -> tuple:
        """
        对交叉验证的一折数据应用PCA降维
        返回: (X_train_processed, X_val_processed, pca_info)
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
        
        # PCA降维
        n_components = actual_dimension
        pca = PCA(n_components=n_components, random_state=42)
        pca_scores_train = pca.fit_transform(back_train_scaled)
        pca_scores_val = pca.transform(back_val_scaled)
        
        # 创建PCA特征
        pca_columns = [f'PCA_主成分{i+1}' for i in range(n_components)]
        pca_train_df = pd.DataFrame(pca_scores_train, columns=pca_columns, index=X_train.index)
        pca_val_df = pd.DataFrame(pca_scores_val, columns=pca_columns, index=X_val.index)
        
        # 合并特征
        if front_feature_count > 0:
            front_train = front_train.reset_index(drop=True)
            front_val = front_val.reset_index(drop=True)
            pca_train_df = pca_train_df.reset_index(drop=True)
            pca_val_df = pca_val_df.reset_index(drop=True)
            
            final_train = pd.concat([front_train, pca_train_df], axis=1)
            final_val = pd.concat([front_val, pca_val_df], axis=1)
        else:
            final_train = pca_train_df
            final_val = pca_val_df
        
        # 保存降维信息
        pca_info = {
            # 'scaler': scaler,
            'pca': pca,
            'n_components': n_components,
            'pca_columns': pca_columns,
            'front_columns': front_train.columns.tolist() if front_feature_count > 0 else [],
            'reduction_ratio': ratio,
            'actual_dimension': actual_dimension
        }
        
        return final_train, final_val, pca_info
    
    def reduce_full_data(self, output_dir, training_params_dir, X: pd.DataFrame, y=None, ratio: float = 0.5, 
                        property_name: str = None) -> tuple:
        """对完整数据集应用PCA降维"""
        # 分离特征
        front_features, back_features = self.split_features(X)
        front_feature_count = len(front_features)
        self.pca_summary_file = f"{output_dir}/pca_summary.json"
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
        
        # PCA降维
        n_components = actual_dimension
        pca = PCA(n_components=n_components, random_state=42)
        pca_scores = pca.fit_transform(back_data_scaled)
        
        # 创建PCA特征
        pca_columns = [f'PCA_主成分{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_scores, columns=pca_columns, index=X.index)
        
        # 合并特征
        if front_feature_count > 0:
            front_data = front_data.reset_index(drop=True)
            pca_df = pca_df.reset_index(drop=True)
            final_data = pd.concat([front_data, pca_df], axis=1)
        else:
            final_data = pca_df
        
        # 计算主成分与y的相关系数
        pca_correlations = {}
        if y is not None:
            for i, col in enumerate(pca_columns):
                correlation = np.corrcoef(pca_scores[:, i], y)[0, 1]
                pca_correlations[col] = correlation
        
        # 保存降维信息
        pca_info = {
            # 'scaler': scaler,
            'pca': pca,
            'n_components': n_components,
            'pca_columns': pca_columns,
            'front_columns': front_data.columns.tolist() if front_feature_count > 0 else [],
            'back_columns': back_features,
            'reduction_ratio': ratio,
            'actual_dimension': actual_dimension,
            'back_feature_count': len(back_features),
            'pca_correlations': pca_correlations,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'explained_variance': pca.explained_variance_.tolist(),
            'cumulative_explained_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pca.components_,  # 变换矩阵
            'mean': pca.mean_,
            'singular_values': pca.singular_values_.tolist(),
            'noise_variance': pca.noise_variance_
        }
        
        # 保存到文件
        if property_name:
            self._save_pca_info(training_params_dir, pca_info, property_name)
            # 生成PCA详细报告
            self._save_pca_detailed_report(pca_info, property_name, X, y)
        
        return final_data, pca_info
    
    def _save_pca_info(self, training_params_dir, pca_info: dict, property_name: str):
        """保存PCA信息到文件"""
        try:
            training_params_dir = training_params_dir
            os.makedirs(training_params_dir, exist_ok=True) 

            # 优化字符替换：用字典批量替换，更简洁易扩展
            replace_map = {"（": "_", "）": "_", "/": "_"}
            safe_prop = property_name
            for old, new in replace_map.items():
                safe_prop = safe_prop.replace(old, new)

            filename = f"{training_params_dir}/pca_params_{safe_prop}.pkl"
            
            with open(filename, 'wb') as f:
                pickle.dump(pca_info, f)
            
            print(f"PCA参数已保存到 {filename}")
        except Exception as e:
            print(f"保存PCA参数失败: {e}")
    
    def _save_pca_detailed_report(self, pca_info: dict, property_name: str, X: pd.DataFrame, y=None):
        """保存PCA详细报告到JSON文件"""
        try:
            summary_file = self.pca_summary_file

            prop_data = self._build_pca_property_data(pca_info, property_name, X, y)

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

            print(f"PCA详细报告已保存到 {summary_file} 的 [{property_name}] 条目中")

        except Exception as e:
            print(f"保存PCA详细报告到JSON时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _build_pca_property_data(self, pca_info: dict, property_name: str, X: pd.DataFrame, y=None):
        """构建单个性质的PCA数据字典"""
        prop_data = {}

        noise_var = pca_info.get('noise_variance')
        if isinstance(noise_var, float) and np.isnan(noise_var):
            noise_var = None

        basic_info = {
            '属性名称': property_name,
            '降维类型': 'PCA',
            '原始后部特征数量': pca_info['back_feature_count'],
            'PCA降维比例': pca_info['reduction_ratio'],
            '实际PCA维度': pca_info['actual_dimension'],
            '噪声方差': noise_var,
            '总方差': float(sum(pca_info['explained_variance'])),
            '保留方差比例': float(sum(pca_info['explained_variance_ratio']))
        }
        prop_data['basic_info'] = basic_info

        variance_data = []
        for i, (var_ratio, cum_ratio) in enumerate(zip(
            pca_info['explained_variance_ratio'],
            pca_info['cumulative_explained_variance_ratio']
        )):
            pca_col = f'PCA_主成分{i+1}'
            correlation = pca_info['pca_correlations'].get(pca_col) if pca_info.get('pca_correlations') else None
            if isinstance(correlation, float) and np.isnan(correlation):
                correlation = None

            sv = pca_info['singular_values'][i] if i < len(pca_info['singular_values']) else None
            if isinstance(sv, float) and np.isnan(sv):
                sv = None
            ev = pca_info['explained_variance'][i] if i < len(pca_info['explained_variance']) else None
            if isinstance(ev, float) and np.isnan(ev):
                ev = None

            variance_data.append({
                '主成分': f'PC{i+1}',
                '特征值': sv,
                '方差解释率': float(var_ratio) * 100,
                '累计方差解释率': float(cum_ratio) * 100,
                '方差贡献': ev,
                '与y的相关系数': correlation
            })
        prop_data['variance'] = variance_data

        loadings_data = []
        full_loadings = None
        if 'components' in pca_info and 'back_columns' in pca_info:
            components = pca_info['components']
            back_columns = pca_info['back_columns']

            if components is not None and len(components) > 0 and back_columns:
                n_components = components.shape[0]
                n_top_features = min(10, len(back_columns))

                for pc_idx in range(n_components):
                    pc_loadings = components[pc_idx]
                    abs_loadings = np.abs(pc_loadings)
                    top_indices = np.argsort(abs_loadings)[-n_top_features:][::-1]

                    for rank, idx in enumerate(top_indices):
                        lv = float(pc_loadings[idx])
                        alv = float(abs_loadings[idx])
                        loadings_data.append({
                            '主成分': f'PC{pc_idx+1}',
                            '排名': rank + 1,
                            '特征名称': str(back_columns[idx]),
                            '载荷值': lv,
                            '载荷绝对值': alv
                        })

                if len(back_columns) <= 50:
                    full_matrix = []
                    for pc_idx in range(n_components):
                        row_vals = [float(v) if not (isinstance(v, float) and np.isnan(v)) else None for v in components[pc_idx]]
                        full_matrix.append(row_vals)
                    full_loadings = {
                        'columns': list(back_columns),
                        'index': [f'PC{i+1}' for i in range(n_components)],
                        'data': full_matrix
                    }

        prop_data['loadings'] = loadings_data
        if full_loadings is not None:
            prop_data['full_loadings'] = full_loadings

        return prop_data
    
    def get_pca_summary_statistics(self, pca_info: dict) -> dict:
        """获取PCA汇总统计信息"""
        summary = {
            'property_name': pca_info.get('property_name', 'Unknown'),
            'n_components': pca_info['n_components'],
            'original_features': pca_info['back_feature_count'],
            'reduction_ratio': pca_info['reduction_ratio'],
            'total_variance_explained': sum(pca_info['explained_variance_ratio']) * 100,
            'first_pc_variance': pca_info['explained_variance_ratio'][0] * 100 if pca_info['explained_variance_ratio'] else 0,
            'noise_variance': pca_info.get('noise_variance', 0),
            'has_correlations': bool(pca_info.get('pca_correlations'))
        }
        
        # 如果有相关系数，添加相关系数信息
        if pca_info.get('pca_correlations'):
            correlations = list(pca_info['pca_correlations'].values())
            summary['max_correlation'] = max(correlations, key=abs)
            summary['avg_abs_correlation'] = np.mean([abs(c) for c in correlations])
        
        return summary
    
    def create_visualization_data(self, pca_info: dict, X: pd.DataFrame, y=None) -> dict:
        """创建可视化数据"""
        viz_data = {}
        
        # 1. 碎石图数据
        viz_data['scree_plot'] = {
            'components': [f'PC{i+1}' for i in range(pca_info['n_components'])],
            'explained_variance_ratio': pca_info['explained_variance_ratio'],
            'cumulative_explained_variance_ratio': pca_info['cumulative_explained_variance_ratio']
        }
        
        # 2. 载荷图数据（前两个主成分）
        if pca_info['n_components'] >= 2:
            viz_data['loadings_plot'] = {
                'features': pca_info['back_columns'],
                'pc1_loadings': pca_info['components'][0, :].tolist() if len(pca_info['components']) > 0 else [],
                'pc2_loadings': pca_info['components'][1, :].tolist() if len(pca_info['components']) > 1 else []
            }
        
        # 3. 主成分得分数据
        if X is not None and pca_info.get('back_columns'):
            back_data = X[pca_info['back_columns']].values.astype(np.float64)
            # scaler = StandardScaler()s
            # back_data_scaled = scaler.fit_transform(back_data)
            back_data_scaled = back_data
            pca_scores = pca_info['pca'].transform(back_data_scaled)
            
            viz_data['scores_plot'] = {
                'samples': list(range(len(pca_scores))),
                'pc1_scores': pca_scores[:, 0].tolist() if pca_info['n_components'] >= 1 else [],
                'pc2_scores': pca_scores[:, 1].tolist() if pca_info['n_components'] >= 2 else []
            }
            
            if y is not None:
                viz_data['scores_plot']['y_values'] = y.tolist()
        
        return viz_data
