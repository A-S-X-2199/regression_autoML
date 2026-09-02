import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
import pickle
import os
import json
from .feature_schema import split_feature_columns

class PLSReducer:
    """PLS降维器"""

    def __init__(self, n_features_for_reduction: int = 50):
        self.n_features_for_reduction = n_features_for_reduction
        self.reduction_type = 'pls'
        # self.pls_summary_file = 'pls_summary.xlsx'

    def split_features(self, X: pd.DataFrame) -> tuple:
        """按输入中的显式标记拆分前部分与参与降维的后部分。"""
        return split_feature_columns(X.columns)

    def reduce_fold(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                   y_train=None, ratio: float = 0.5) -> tuple:
        """
        对交叉验证的一折数据应用PLS降维
        返回: (X_train_processed, X_val_processed, pls_info)
        """
        if y_train is None:
            raise ValueError("PLS降维需要目标变量y_train")

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

        # PLS降维
        n_components = actual_dimension
        pls = PLSRegression(n_components=n_components)
        pls.fit(back_train_scaled, y_train)

        # 转换数据
        pls_scores_train = pls.transform(back_train_scaled)
        pls_scores_val = pls.transform(back_val_scaled)

        # 创建PLS特征
        pls_columns = [f'PLS_主成分{i+1}' for i in range(n_components)]
        pls_train_df = pd.DataFrame(pls_scores_train, columns=pls_columns, index=X_train.index)
        pls_val_df = pd.DataFrame(pls_scores_val, columns=pls_columns, index=X_val.index)

        # 合并特征
        if front_feature_count > 0:
            front_train = front_train.reset_index(drop=True)
            front_val = front_val.reset_index(drop=True)
            pls_train_df = pls_train_df.reset_index(drop=True)
            pls_val_df = pls_val_df.reset_index(drop=True)

            final_train = pd.concat([front_train, pls_train_df], axis=1)
            final_val = pd.concat([front_val, pls_val_df], axis=1)
        else:
            final_train = pls_train_df
            final_val = pls_val_df

        # 保存降维信息
        pls_info = {
            # 'scaler': scaler,
            'pls': pls,
            'n_components': n_components,
            'pls_columns': pls_columns,
            'front_columns': front_train.columns.tolist() if front_feature_count > 0 else [],
            'reduction_ratio': ratio,
            'actual_dimension': actual_dimension
        }

        return final_train, final_val, pls_info

    def reduce_full_data(self, output_dir, training_params_dir, X: pd.DataFrame, y=None, ratio: float = 0.5, 
                        property_name: str = None) -> tuple:
        """对完整数据集应用PLS降维"""
        if y is None:
            raise ValueError("PLS降维需要目标变量y")
        self.pls_summary_file = f"{output_dir}/pls_summary.json"
        # 分离特征
        front_features, back_features = self.split_features(X)
        front_feature_count = len(front_features)

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

        # PLS降维
        n_components = actual_dimension
        pls = PLSRegression(n_components=n_components)
        pls.fit(back_data_scaled, y)

        # 转换数据
        pls_scores = pls.transform(back_data_scaled)

        # 创建PLS特征
        pls_columns = [f'PLS_主成分{i+1}' for i in range(n_components)]
        pls_df = pd.DataFrame(pls_scores, columns=pls_columns, index=X.index)

        # 合并特征
        if front_feature_count > 0:
            front_data = front_data.reset_index(drop=True)
            pls_df = pls_df.reset_index(drop=True)
            final_data = pd.concat([front_data, pls_df], axis=1)
        else:
            final_data = pls_df

        # 计算主成分与y的相关系数
        pls_correlations = {}
        for i, col in enumerate(pls_columns):
            correlation = np.corrcoef(pls_scores[:, i], y)[0, 1]
            pls_correlations[col] = correlation

        # 保存降维信息
        pls_info = {
            # 'scaler': scaler,
            'pls': pls,
            'n_components': n_components,
            'pls_columns': pls_columns,
            'front_columns': front_data.columns.tolist() if front_feature_count > 0 else [],
            'back_columns': back_features,
            'reduction_ratio': ratio,
            'actual_dimension': actual_dimension,
            'back_feature_count': len(back_features),
            'pls_correlations': pls_correlations,
            'x_weights': pls.x_weights_.T.tolist(), 
            'y_weights': pls.y_weights_.tolist() if hasattr(pls, 'y_weights_') else [],
            'x_loadings': pls.x_loadings_.tolist() if hasattr(pls, 'x_loadings_') else [],
            'coefficients': pls.coef_.tolist() if hasattr(pls, 'coef_') else [],
            'x_scores': pls.x_scores_.tolist() if hasattr(pls, 'x_scores_') else []
        }

        # 保存到文件
        if property_name:
            self._save_pls_info(training_params_dir, pls_info, property_name)
            # 生成PLS详细报告
            self._save_pls_detailed_report(pls_info, property_name, X, y)

        return final_data, pls_info

    def _save_pls_info(self, training_params_dir, pls_info: dict, property_name: str):
        """保存PLS信息到文件"""
        try:
            training_params_dir = training_params_dir
            os.makedirs(training_params_dir, exist_ok=True) 

            # 优化字符替换：用字典批量替换，更简洁易扩展
            replace_map = {"（": "_", "）": "_", "/": "_"}
            safe_prop = property_name
            for old, new in replace_map.items():
                safe_prop = safe_prop.replace(old, new)

            filename = f"{training_params_dir}/pls_params_{safe_prop}.pkl"

            with open(filename, 'wb') as f:
                pickle.dump(pls_info, f)

            print(f"PLS参数已保存到 {filename}")
        except Exception as e:
            print(f"保存PLS参数失败: {e}")

    def _save_pls_detailed_report(self, pls_info: dict, property_name: str, X: pd.DataFrame, y=None):
        """保存PLS详细报告到JSON文件"""
        try:
            summary_file = self.pls_summary_file

            prop_data = self._build_pls_property_data(pls_info, property_name, X, y)

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

            print(f"PLS详细报告已保存到 {summary_file} 的 [{property_name}] 条目中")

        except Exception as e:
            print(f"保存PLS详细报告到JSON时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _build_pls_property_data(self, pls_info: dict, property_name: str, X: pd.DataFrame, y=None):
        """构建单个性质的PLS数据字典"""
        prop_data = {}

        basic_info = {
            '属性名称': property_name,
            '降维类型': 'PLS',
            '原始后部特征数量': pls_info['back_feature_count'],
            'PLS降维比例': pls_info['reduction_ratio'],
            '实际PLS维度': pls_info['actual_dimension']
        }
        prop_data['basic_info'] = basic_info

        correlation_data = []
        if pls_info['pls_correlations'] and len(pls_info['pls_correlations']) > 0:
            for pc, corr in pls_info['pls_correlations'].items():
                c = float(corr) if not (isinstance(corr, float) and np.isnan(corr)) else None
                correlation_data.append({
                    '主成分': pc,
                    '与目标性质相关系数': c,
                    '相关系数绝对值': abs(c) if c is not None else None
                })
            correlation_data.sort(key=lambda x: abs(x['与目标性质相关系数']) if x['与目标性质相关系数'] is not None else 0, reverse=True)
        prop_data['correlation'] = correlation_data

        weights_data = []
        full_weights = None
        if 'x_weights' in pls_info and pls_info['x_weights']:
            x_weights_array = np.array(pls_info['x_weights'])

            if x_weights_array.ndim == 2:
                if x_weights_array.shape[0] > x_weights_array.shape[1]:
                    x_weights_array = x_weights_array.T

                n_components = x_weights_array.shape[0]
                n_features = x_weights_array.shape[1]

                n_top_features = min(10, n_features)

                for pc_idx in range(n_components):
                    pc_weights = x_weights_array[pc_idx]
                    abs_weights = np.abs(pc_weights)
                    top_indices = np.argsort(abs_weights)[-n_top_features:][::-1]

                    for rank, idx in enumerate(top_indices):
                        feature_name = None
                        if 'back_columns' in pls_info and len(pls_info['back_columns']) > idx:
                            feature_name = str(pls_info['back_columns'][idx])
                        else:
                            feature_name = f'特征{idx+1}'

                        wv = float(pc_weights[idx])
                        awv = float(abs_weights[idx])
                        weights_data.append({
                            '主成分': f'PLS{pc_idx+1}',
                            '排名': rank + 1,
                            '特征名称': feature_name,
                            '权重值': wv,
                            '权重绝对值': awv
                        })

                if 'back_columns' in pls_info:
                    back_columns = pls_info['back_columns']
                    if len(back_columns) <= 50:
                        full_matrix = []
                        for pc_idx in range(n_components):
                            row_vals = [float(v) if not (isinstance(v, float) and np.isnan(v)) else None for v in x_weights_array[pc_idx]]
                            full_matrix.append(row_vals)
                        full_weights = {
                            'columns': list(back_columns),
                            'index': [f'PLS{i+1}' for i in range(n_components)],
                            'data': full_matrix
                        }

        prop_data['weights'] = weights_data
        if full_weights is not None:
            prop_data['full_weights'] = full_weights

        return prop_data

    def get_pls_summary_statistics(self, pls_info: dict) -> dict:
        """获取PLS汇总统计信息"""
        summary = {
            'property_name': pls_info.get('property_name', 'Unknown'),
            'n_components': pls_info['n_components'],
            'original_features': pls_info['back_feature_count'],
            'reduction_ratio': pls_info['reduction_ratio'],
            'has_correlations': bool(pls_info.get('pls_correlations'))
        }

        # 如果有相关系数，添加相关系数信息
        if pls_info.get('pls_correlations'):
            correlations = list(pls_info['pls_correlations'].values())
            summary['max_correlation'] = max(correlations, key=abs)
            summary['avg_abs_correlation'] = np.mean([abs(c) for c in correlations])

        return summary

    def create_visualization_data(self, pls_info: dict, X: pd.DataFrame, y=None) -> dict:
        """创建可视化数据"""
        viz_data = {}

        # 1. 相关系数数据
        if pls_info.get('pls_correlations'):
            viz_data['correlation_plot'] = {
                'components': list(pls_info['pls_correlations'].keys()),
                'correlations': list(pls_info['pls_correlations'].values()),
                'abs_correlations': [abs(c) for c in pls_info['pls_correlations'].values()]
            }

        # 2. 权重图数据（前两个主成分）
        if pls_info['n_components'] >= 2 and 'x_weights' in pls_info:
            x_weights_array = np.array(pls_info['x_weights'])
            
            # 确保我们有正确的形状 [n_components, n_features]
            if x_weights_array.ndim == 2:
                # 如果是 [n_features, n_components] 形状，转置它
                if x_weights_array.shape[0] > x_weights_array.shape[1]:
                    x_weights_array = x_weights_array.T
                    
                if x_weights_array.shape[0] >= 2:
                    viz_data['weights_plot'] = {
                        'features': pls_info.get('back_columns', [f'特征{i+1}' for i in range(x_weights_array.shape[1])]),
                        'pls1_x_weights': x_weights_array[0, :].tolist(),
                        'pls2_x_weights': x_weights_array[1, :].tolist()
                    }

        # 3. PLS得分图数据
        if 'x_scores' in pls_info and len(pls_info['x_scores']) > 0:
            x_scores_array = np.array(pls_info['x_scores'])
            if x_scores_array.ndim == 2:
                viz_data['scores_plot'] = {
                    'samples': list(range(len(x_scores_array))),
                    'pls1_scores': x_scores_array[:, 0].tolist() if pls_info['n_components'] >= 1 else [],
                    'pls2_scores': x_scores_array[:, 1].tolist() if pls_info['n_components'] >= 2 else []
                }

                if y is not None:
                    viz_data['scores_plot']['y_values'] = y.tolist()

        return viz_data
    
    def load_pls_info(self, filepath: str) -> dict:
        """加载PLS信息"""
        try:
            with open(filepath, 'rb') as f:
                pls_info = pickle.load(f)
            print(f"已从 {filepath} 加载PLS参数")
            return pls_info
        except Exception as e:
            print(f"加载PLS参数失败: {e}")
            return None

    def apply_pls_transform(self, X: pd.DataFrame, pls_info: dict) -> pd.DataFrame:
        """应用已训练的PLS转换"""
        # 分离特征
        front_feature_count = len(pls_info.get('front_columns', []))
        
        # 前部分特征
        front_data = X.iloc[:, :front_feature_count] if front_feature_count > 0 else pd.DataFrame()
        
        # 后部分特征
        back_data = X.iloc[:, front_feature_count:].values.astype(np.float64)
        
        if back_data.shape[1] == 0:
            return front_data
        
        # 标准化
        # scaler = pls_info['scaler']
        # back_data_scaled = scaler.transform(back_data)
        back_data_scaled = back_data
        
        # PLS转换
        pls = pls_info['pls']
        pls_scores = pls.transform(back_data_scaled)
        
        # 创建PLS特征
        pls_df = pd.DataFrame(pls_scores, columns=pls_info.get('pls_columns', [f'PLS_主成分{i+1}' for i in range(pls_scores.shape[1])]), 
                            index=X.index)
        
        # 合并特征
        if front_feature_count > 0:
            front_data = front_data.reset_index(drop=True)
            pls_df = pls_df.reset_index(drop=True)
            final_data = pd.concat([front_data, pls_df], axis=1)
        else:
            final_data = pls_df
            
        return final_data

    def compare_pls_models(self, pls_info_list: list, model_names: list = None) -> pd.DataFrame:
        """比较多个PLS模型"""
        if model_names is None:
            model_names = [f'Model_{i+1}' for i in range(len(pls_info_list))]

        comparison_data = []

        for i, (pls_info, name) in enumerate(zip(pls_info_list, model_names)):
            summary = self.get_pls_summary_statistics(pls_info)
            summary['model_name'] = name
            comparison_data.append(summary)

        comparison_df = pd.DataFrame(comparison_data)

        try:
            comparison_file = "pls_model_comparison.json"
            comparison_dict = {
                'model_comparison': comparison_df.to_dict(orient='records'),
                'model_details': []
            }

            for i, (pls_info, name) in enumerate(zip(pls_info_list, model_names)):
                detail = {
                    'model_name': name,
                    '属性': pls_info.get('property_name', 'Unknown'),
                    '主成分数量': pls_info['n_components'],
                    '原始特征数': pls_info['back_feature_count'],
                    '降维比例': pls_info['reduction_ratio']
                }
                comparison_dict['model_details'].append(detail)

            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_dict, f, ensure_ascii=False, indent=2, default=str)

            print(f"PLS模型比较结果已保存到 {comparison_file}")
        except Exception as e:
            print(f"保存比较结果时出错: {e}")

        return comparison_df
