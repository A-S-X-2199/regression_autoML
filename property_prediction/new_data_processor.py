import pandas as pd
import numpy as np
import pickle
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class NewDataProcessor:
    def __init__(self, predict=False,
                 best_models_dir='best_models',
                 output_path='test_data_processed.xlsx',
                 training_params_dir='training_params_dir',
                 standardization_params_path='standardization_params_.pkl',
                 new_data_df=None,
                 property_names=None,
                 final_models_dir=None):
        self.predict = predict
        self.best_models_dir = best_models_dir
        self.output_path = output_path
        self.training_params_dir = training_params_dir
        self.standardization_params_path = standardization_params_path
        self.new_data_df = new_data_df
        self.property_names = property_names
        self.final_models_dir = final_models_dir if final_models_dir else None
        self.best_models_info = {}
        self.reduction_params = {}
        self.new_df = None
        self.new_df_original = None
        self.new_df_standardized = None
        self.new_sample_matrix = None
        self.new_features_dict = {}
        self.final_results_dict = {}
        self.scaler_params = None

    def _load_best_models_info(self):
        """加载每个目标的最佳模型信息。"""
        print("\n=== 1. 加载最佳模型信息 ===")
        try:
            if not os.path.exists(self.best_models_dir):
                raise FileNotFoundError(f"最佳模型目录不存在: {self.best_models_dir}")
            
            info_files = [f for f in os.listdir(self.best_models_dir) 
                         if f.startswith('best_model_info_') and f.endswith('.json')]
            
            if not info_files:
                raise FileNotFoundError(f"在 {self.best_models_dir} 中未找到最佳模型信息文件")
            
            for info_file in info_files:
                try:
                    prop_name = info_file.replace('best_model_info_', '').replace('.json', '')
                    if self.property_names is not None and prop_name not in self.property_names:
                        continue
                    with open(os.path.join(self.best_models_dir, info_file), 'r', encoding='utf-8') as f:
                        model_info = json.load(f)
                    self.best_models_info[prop_name] = model_info
                    reduction_type = model_info.get('reduction_type', 'none')
                    print(f"加载性质 '{prop_name}' 的最佳模型:")
                    print(f"  - 模型类型: {model_info['best_model']}")
                    print(f"  - 降维方式: {reduction_type.upper()}")
                    
                    self._load_reduction_params(prop_name, reduction_type)
                except Exception as e:
                    print(f"加载模型信息文件 {info_file} 失败: {e}")
                    continue
            
            print(f"成功加载 {len(self.best_models_info)} 个性质的最佳模型信息（各性质独立配置）")
            return True
            
        except Exception as e:
            print(f"加载最佳模型信息失败: {str(e)}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return False
    
    def _find_param_file(self, filename: str) -> str:
        """查找参数文件路径：优先 final_models_dir（全量数据最终模型），回退 training_params_dir"""
        if self.final_models_dir:
            cand = os.path.join(self.final_models_dir, filename)
            if os.path.exists(cand):
                return cand
        return os.path.join(self.training_params_dir, filename)

    def _load_reduction_params(self, prop_name, reduction_type):
        """加载每个性质的独立降维参数（优先从 final_models_dir，回退 training_params_dir）"""
        safe_prop = prop_name.replace("（", "_").replace("）", "_").replace("/", "_")

        if reduction_type == 'pca':
            pca_file = self._find_param_file(f"pca_params_{safe_prop}.pkl")
            if os.path.exists(pca_file):
                with open(pca_file, 'rb') as f:
                    self.reduction_params[prop_name] = {'type': 'pca', 'params': pickle.load(f)}
                print(f"    加载PCA参数成功: {pca_file}")
            else:
                print(f"    警告: 未找到PCA参数文件 {pca_file}，使用无降维")
                self.reduction_params[prop_name] = {'type': 'none'}
                
        elif reduction_type == 'pls':
            pls_file = self._find_param_file(f"pls_params_{safe_prop}.pkl")
            if os.path.exists(pls_file):
                with open(pls_file, 'rb') as f:
                    self.reduction_params[prop_name] = {'type': 'pls', 'params': pickle.load(f)}
                print(f"    加载PLS参数成功: {pls_file}")
            else:
                print(f"    警告: 未找到PLS参数文件 {pls_file}，使用无降维")
                self.reduction_params[prop_name] = {'type': 'none'}
        
        elif reduction_type == 'svd':
            svd_file = self._find_param_file(f"svd_params_{safe_prop}.pkl")
            if os.path.exists(svd_file):
                with open(svd_file, 'rb') as f:
                    self.reduction_params[prop_name] = {'type': 'svd', 'params': pickle.load(f)}
                print(f"    加载SVD参数成功: {svd_file}")
            else:
                print(f"    警告: 未找到SVD参数文件 {svd_file}，使用无降维")
                self.reduction_params[prop_name] = {'type': 'none'}
        
        elif reduction_type == 'tsvd':
            tsvd_file = self._find_param_file(f"tsvd_params_{safe_prop}.pkl")
            if os.path.exists(tsvd_file):
                with open(tsvd_file, 'rb') as f:
                    self.reduction_params[prop_name] = {'type': 'tsvd', 'params': pickle.load(f)}
                print(f"    加载tSVD参数成功: {tsvd_file}")
            else:
                svd_file = self._find_param_file(f"svd_params_{safe_prop}.pkl")
                if os.path.exists(svd_file):
                    with open(svd_file, 'rb') as f:
                        self.reduction_params[prop_name] = {'type': 'tsvd', 'params': pickle.load(f)}
                    print(f"    使用SVD参数作为tSVD参数: {svd_file}")
                else:
                    print(f"    警告: 未找到tSVD/SVD参数文件，使用无降维")
                    self.reduction_params[prop_name] = {'type': 'none'}
        else:
            self.reduction_params[prop_name] = {'type': 'none'}

    def load_new_data(self):
        """加载新数据并保留完整的前、后部分特征。"""
        print("\n=== 2. 加载新数据 ===")
        try:
            if self.new_data_df is not None:
                self.new_df_original = self.new_data_df.copy()
                id_col_name = self.new_df_original.columns[0]
                self.new_df = self.new_df_original.copy().set_index(id_col_name)
                self.new_sample_matrix = self.new_df.values.astype(np.float64)
                print(f"加载新数据: 从DataFrame传入，形状: {self.new_df.shape}")
            else:
                raise ValueError("未提供新数据DataFrame new_data_df")
            
            print(f"  - 样本数: {len(self.new_df)}")
            print(f"  - 原始特征数: {len(self.new_df.columns)}")
            print(f"  - 索引列: '{id_col_name}'")
            
            return True
        except Exception as e:
            print(f"加载新数据失败: {str(e)}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return False

    def standardize_new_data(self):
        """保留原始输入；标准化统一在后续按目标使用训练参数完成。"""
        print("\n=== 3. 标准化新数据 ===")
        self.new_df_standardized = None
        self.scaler_params = None
        print("新数据将在每个目标下按训练列对齐，缺失特征补 0 后独立标准化")
        return True

    def calculate_features_for_all_properties(self):
        """为所有目标独立执行降维、特征提取和目标级标准化。"""
        print("\n=== 4. 为所有性质独立处理特征 ===")
        # 优先使用标准化后的数据，否则使用原始数据
        if self.new_df_standardized is not None:
            base_df = self.new_df_standardized.copy()
            print("使用全局标准化数据作为基础（将按性质加载逐性质标准化参数覆盖）")
        elif self.new_df is not None:
            base_df = self.new_df.copy()
            print("使用原始数据（未标准化）作为基础")
        else:
            print("错误: 未加载任何数据，无法处理特征")
            return False
        
        success_count = 0
        total_properties = len(self.best_models_info)
        
        for prop_name, model_info in self.best_models_info.items():
            print(f"\n--- 处理性质: {prop_name} ---")
            # 1. 获取当前目标的降维方式
            reduction_info = self.reduction_params.get(prop_name, {'type': 'none'})
            reduction_type = reduction_info['type']
            
            try:
                # 2. 逐性质标准化：尝试加载该性质的独立标准化参数（优先 final_models_dir）
                safe_prop = prop_name.replace("（", "_").replace("）", "_").replace("/", "_")
                prop_scaler_path = self._find_param_file(
                    f"standardization_params_{safe_prop}.pkl"
                )
                
                if os.path.exists(prop_scaler_path):
                    # 加载逐性质标准化参数并应用
                    with open(prop_scaler_path, 'rb') as f:
                        prop_scaler_params = pickle.load(f)
                    scaler = StandardScaler()
                    scaler.mean_ = prop_scaler_params['mean']
                    scaler.scale_ = prop_scaler_params['std']
                    scaler_feature_names = prop_scaler_params['feature_names']
                    
                    # 严格按训练列对齐；预测缺少任一前/后部分特征时按0补齐。
                    if scaler_feature_names:
                        missing_raw = [c for c in scaler_feature_names if c not in base_df.columns]
                        aligned_df = base_df.reindex(columns=scaler_feature_names, fill_value=0.0)
                        raw_features = aligned_df.values.astype(np.float64)
                        scaled_features = scaler.transform(raw_features)
                        prop_feature_df = pd.DataFrame(
                            scaled_features,
                            columns=scaler_feature_names,
                            index=base_df.index
                        )
                        print(f"  逐性质标准化: 加载参数成功, 特征形状={prop_feature_df.shape}")
                        if missing_raw:
                            print(f"    预测输入缺失特征已按0补齐: {len(missing_raw)} 个")
                        print(f"    训练集样本数: {prop_scaler_params.get('fitted_on_train_size', 'N/A')}")
                    else:
                        print(f"  警告: 标准化参数特征与当前数据无交集，使用基础数据")
                        prop_feature_df = base_df.copy()
                else:
                    print(f"  未找到逐性质标准化参数({prop_scaler_path})，使用基础数据")
                    prop_feature_df = base_df.copy()
                
                print(f"  初始特征形状: {prop_feature_df.shape}")
                
                # 3. 模型需要的特征补齐（预测时可能缺少部分特征，用0填充）
                model_features = model_info.get('features', [])
                missing_in_data = [c for c in model_features if c not in prop_feature_df.columns]
                if missing_in_data:
                    for col in missing_in_data:
                        prop_feature_df[col] = 0.0
                    print(f"  补齐缺失特征 {len(missing_in_data)} 个: {missing_in_data[:5]}...")

                # 4. 根据当前性质的降维方式，独立处理特征
                if reduction_type == 'pca':
                    features = self._calculate_features_pca_mode(prop_name, reduction_info['params'], prop_feature_df)
                elif reduction_type == 'pls':
                    features = self._calculate_features_pls_mode(prop_name, reduction_info['params'], prop_feature_df)
                elif reduction_type in ['svd', 'tsvd']:
                    features = self._calculate_features_svd_mode(prop_name, reduction_info['params'], reduction_type, prop_feature_df)
                else:
                    features = self._calculate_features_no_reduction(prop_name, model_info, prop_feature_df)
                
                if features is not None:
                    self.new_features_dict[prop_name] = features
                    success_count += 1
                    print(f"性质 '{prop_name}' 特征处理成功: 最终形状 {features.shape}")
                else:
                    print(f"性质 '{prop_name}' 特征处理失败")
            except Exception as e:
                print(f"性质 '{prop_name}' 特征处理失败: {str(e)}")
                import traceback
                print(f"详细错误: {traceback.format_exc()}")
                continue
        
        print(f"\n所有性质特征处理完成: {success_count}/{total_properties} 个性质成功（各性质独立处理，无相互干扰）")
        return success_count > 0
    
    def _calculate_features_pca_mode(self, prop_name, pca_params, prop_feature_df):
        """PCA模式：基于当前性质的独立特征副本处理"""
        print(f"  降维方式: PCA")
        pca = pca_params['pca']
        n_components = pca_params['n_components']
        front_columns = pca_params.get('front_columns', [])
        back_columns = pca_params.get('back_columns', [])

        front_feat = prop_feature_df.reindex(columns=front_columns, fill_value=0.0).values if front_columns else np.empty((len(prop_feature_df), 0))
        back_feat = prop_feature_df.reindex(columns=back_columns, fill_value=0.0).values if back_columns else np.empty((len(prop_feature_df), 0))
        pca_trans = pca.transform(back_feat) if back_feat.shape[1] > 0 else np.empty((len(prop_feature_df), 0))

        pca_df = pd.DataFrame(pca_trans, columns=[f'PCA_主成分{i+1}' for i in range(n_components)], index=prop_feature_df.index)
        front_df = pd.DataFrame(front_feat, columns=front_columns, index=prop_feature_df.index) if front_columns else pd.DataFrame(index=prop_feature_df.index)
        return pd.concat([front_df, pca_df], axis=1)
    
    def _calculate_features_pls_mode(self, prop_name, pls_params, prop_feature_df):
        """PLS模式：基于当前性质的独立特征副本处理"""
        print(f"  降维方式: PLS")
        pls = pls_params['pls']
        n_components = pls_params['n_components']
        front_columns = pls_params.get('front_columns', [])
        back_columns = pls_params.get('back_columns', [])

        front_feat = prop_feature_df.reindex(columns=front_columns, fill_value=0.0).values if front_columns else np.empty((len(prop_feature_df), 0))
        back_feat = prop_feature_df.reindex(columns=back_columns, fill_value=0.0).values if back_columns else np.empty((len(prop_feature_df), 0))
        pls_trans = pls.transform(back_feat) if back_feat.shape[1] > 0 else np.empty((len(prop_feature_df), 0))

        pls_df = pd.DataFrame(pls_trans, columns=[f'PLS_主成分{i+1}' for i in range(n_components)], index=prop_feature_df.index)
        front_df = pd.DataFrame(front_feat, columns=front_columns, index=prop_feature_df.index) if front_columns else pd.DataFrame(index=prop_feature_df.index)
        return pd.concat([front_df, pls_df], axis=1)
    
    def _calculate_features_svd_mode(self, prop_name, svd_params, svd_type, prop_feature_df):
        """SVD/tSVD模式：基于当前性质的独立特征副本处理"""
        print(f"  降维方式: {svd_type.upper()}")
        svd = svd_params['svd']
        n_components = svd_params['n_components']
        front_columns = svd_params.get('front_columns', [])
        back_columns = svd_params.get('back_columns', [])

        front_feat = prop_feature_df.reindex(columns=front_columns, fill_value=0.0).values if front_columns else np.empty((len(prop_feature_df), 0))
        back_feat = prop_feature_df.reindex(columns=back_columns, fill_value=0.0).values if back_columns else np.empty((len(prop_feature_df), 0))
        svd_trans = svd.transform(back_feat) if back_feat.shape[1] > 0 else np.empty((len(prop_feature_df), 0))

        svd_df = pd.DataFrame(svd_trans, columns=[f'SVD_主成分{i+1}' for i in range(n_components)], index=prop_feature_df.index)
        front_df = pd.DataFrame(front_feat, columns=front_columns, index=prop_feature_df.index) if front_columns else pd.DataFrame(index=prop_feature_df.index)
        return pd.concat([front_df, svd_df], axis=1)
    
    def _calculate_features_no_reduction(self, prop_name, model_info, prop_feature_df):
        """无降维模式：基于当前性质的独立特征副本提取所需特征"""
        print(f"  降维方式: 无降维")
        safe_prop = prop_name.replace("（", "_").replace("）", "_").replace("/", "_")
        features_file = os.path.join(self.best_models_dir, f'best_features_{safe_prop}.json')

        try:
            with open(features_file, 'r', encoding='utf-8') as f:
                required_features = json.load(f)
        except Exception as e:
            print(f"  错误: 加载特征列表失败: {e}")
            return None

        return prop_feature_df.reindex(columns=required_features, fill_value=0.0).copy()

    def merge_and_save_results(self):
        """保存所有性质的独立处理结果"""
        print("\n=== 5. 保存所有性质的独立处理结果 ===")
        try:
            if not self.new_features_dict:
                raise ValueError("没有成功处理任何性质的特征，无法保存结果")
            
            original_index_name = self.new_df.index.name or '样本编号'
            for prop_name, features in self.new_features_dict.items():
                result_df = features.reset_index().rename(columns={original_index_name: '样本编号'})
                model_info = self.best_models_info[prop_name]
                result_df['数据集类型'] = '测试集'
                result_df['评估目标'] = prop_name
                result_df['是否标准化'] = '是'
                result_df['降维方式'] = self.reduction_params[prop_name]['type'].upper()
                meta_cols = ['样本编号', '数据集类型', '评估目标', '是否标准化', '降维方式']
                feature_cols = [col for col in result_df.columns if col not in meta_cols]
                self.final_results_dict[prop_name] = result_df[meta_cols + feature_cols]
                print(f"  性质 '{prop_name}' 结果表创建完成: {result_df.shape}")
            
            json_output_path = os.path.splitext(self.output_path)[0] + '.json'
            results_json = {}
            for prop_name, result_df in self.final_results_dict.items():
                results_json[prop_name] = {
                    'columns': list(result_df.columns),
                    'data': result_df.to_dict('records')
                }
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, ensure_ascii=False, indent=2)
            
            self._create_summary_json()
            self._create_standardization_info_json()
            
            print(f"\n所有性质处理结果已保存到: {json_output_path}")
            print(f"  - 共处理 {len(self.final_results_dict)} 个性质")
            print(f"  - 每个目标包含独立处理特征和降维元数据")
            return True
        except Exception as e:
            print(f"保存结果失败: {str(e)}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return False

    def _create_summary_json(self):
        """创建处理汇总JSON（记录每个性质的独立配置和处理结果）"""
        summary_data = []
        for prop_name, model_info in self.best_models_info.items():
            features_df = self.final_results_dict.get(prop_name)
            reduction_info = self.reduction_params[prop_name]
            summary_data.append({
                '性质名称': prop_name,
                '最佳模型': model_info.get('best_model', '未知'),
                '降维方式': reduction_info['type'].upper(),
                '测试集样本数': len(features_df) if features_df is not None else 0,
                '最终特征数': len(features_df.columns) - 5 if features_df is not None else 0,
                '训练集R²': model_info.get('train_r2', 'N/A'),
                '测试集得分': model_info.get('test_score', 'N/A')
            })
        summary_path = os.path.splitext(self.output_path)[0] + '_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"  汇总JSON创建完成: {summary_path}")

    def _create_standardization_info_json(self):
        """创建标准化信息JSON"""
        if self.scaler_params is None:
            print("  无标准化参数，跳过创建标准化信息JSON")
            return
        std_info = []
        for i, (f, m, s) in enumerate(zip(self.scaler_params['feature_names'], self.scaler_params['mean'], self.scaler_params['std'])):
            std_info.append({'特征序号': i+1, '特征名称': f, '训练集均值': round(m,6), '训练集标准差': round(s,6)})
        std_info_data = {
            '标准化参数明细': std_info,
            '标准化摘要': {
                '训练集样本数': self.scaler_params['fitted_on_train_size'],
                '标准化特征数': len(self.scaler_params['feature_names']),
                '标准化方式': 'StandardScaler（Z-score）',
                '参数文件': os.path.basename(self.standardization_params_path)
            }
        }
        std_info_path = os.path.splitext(self.output_path)[0] + '_std_info.json'
        with open(std_info_path, 'w', encoding='utf-8') as f:
            json.dump(std_info_data, f, ensure_ascii=False, indent=2)
        print(f"  标准化信息JSON创建完成: {std_info_path}")

    def get_feature_data_for_model(self, property_name):
        """获取指定性质的模型输入特征（独立处理后的特征）"""
        print(f"\n=== 获取性质 '{property_name}' 的模型输入特征 ===")
        if property_name not in self.best_models_info or property_name not in self.final_results_dict:
            print(f"错误: 未找到性质 '{property_name}' 的处理信息/特征数据")
            return None
        
        model_info = self.best_models_info[property_name]
        safe_prop = property_name.replace("（", "_").replace("）", "_").replace("/", "_")
        try:
            with open(os.path.join(self.best_models_dir, f'best_features_{safe_prop}.json'), 'r', encoding='utf-8') as f:
                required_features = json.load(f)
        except Exception as e:
            print(f"加载特征列表失败: {e}")
            return None
        
        # 从当前性质的独立结果中提取特征（排除元数据）
        feature_data_full = self.final_results_dict[property_name]
        meta_cols = ['样本标识', '数据集类型', '原始编号', '评估性质', '是否标准化', '降维方式']
        feature_cols = [col for col in required_features if col in feature_data_full.columns and col not in meta_cols]
        feature_data = feature_data_full[feature_cols]
        
        print(f"  最佳模型: {model_info['best_model']}")
        print(f"  所需特征数: {len(required_features)}，实际提取数: {len(feature_cols)}")
        print(f"  特征数据形状: {feature_data.shape}")
        return feature_data

    def run_pipeline(self):
        """运行完整的测试集处理流程（各性质独立处理）"""
        print("="*80)
        print("开始测试集数据处理流程（各性质按自身配置独立处理）")
        print("="*80)
        
        steps = [
            self._load_best_models_info,
            self.load_new_data,
            self.standardize_new_data,
            self.calculate_features_for_all_properties,
            self.merge_and_save_results
        ]
        
        for i, step in enumerate(steps, 1):
            # 预测模式强制执行标准化，非预测模式按原逻辑
            if i == 3 and not self.predict:
                self.new_df_standardized = self.new_df.copy()
                print(f"\n>>> 跳过第{i}步: {step.__name__}（非预测模式）")
                continue
            print(f"\n>>> 执行第{i}步: {step.__name__}")
            if not step():
                print(f"!!! 第{i}步执行失败，终止处理流程")
                return False
            print(f"<<< 第{i}步执行成功")
        
        print("\n" + "="*80)
        print("测试集数据处理完成！所有性质均按自身训练配置独立处理")
        print("="*80)
        return True


# 使用示例
if __name__ == "__main__":
    print("新数据处理模块已更新为DataFrame直传模式，请通过Excute_pipe.py调用")
