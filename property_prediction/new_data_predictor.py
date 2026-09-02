import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Optional, Union, Tuple
import torch


class MultiPropertyPredictor:
    def __init__(self, predict: bool = False,
                 best_models_dir: str = 'best_models',
                 output_path: str = '新数据目标预测结果.xlsx',
                 json_output_path: str = '',
                 apply_non_negative: bool = False,
                 apply_range_constraint: bool = False,
                 test_features_df=None,
                 train_data_df=None,
                 final_models_dir=None):
        self.predict = predict
        self.best_models_dir = str(best_models_dir)
        self.output_path = str(output_path)
        self.json_output_path = str(json_output_path)
        self.apply_non_negative = apply_non_negative
        self.apply_range_constraint = apply_range_constraint
        self.test_features_df = test_features_df
        self.train_data_df = train_data_df
        self.final_models_dir = str(final_models_dir) if final_models_dir else None
        
        self.best_models_info = {}
        self.models = {}
        self.feature_lists = {}
        self.test_data_dict = {}
        self.predictions_dict = {}
        self.property_ranges = {}
        self.nn_backend_info = {}
        self.property_list = None
        self.reduction_params = {}
        self.prediction_result_df = None  # 保存预测结果的DataFrame
        self._train_test_metrics = None  # 缓存 train_output/output.json 的测试集指标（兜底用）
        
        # PyTorch支持
        self.pytorch_available = False
        try:
            import torch.nn as nn
            self.pytorch_available = True
        except ImportError:
            print("警告: PyTorch不可用，将跳过PyTorch模型加载")
    
    def set_property_list(self, property_list: List[str]):
        """设置要预测的性质列表"""
        self.property_list = property_list
        print(f"设置预测性质列表: {property_list}")
    
    def _find_param_file(self, filename: str) -> Optional[str]:
        """查找参数文件：优先 final_models_dir（全量数据最终模型），回退旧布局（reduction_params/ 与根目录）"""
        if self.final_models_dir:
            cand = os.path.join(self.final_models_dir, filename)
            if os.path.exists(cand):
                return cand
        base_dir = os.path.dirname(self.best_models_dir)
        candidate_files = [
            os.path.join(base_dir, 'reduction_params', filename),
            os.path.join(base_dir, filename),
        ]
        for param_file in candidate_files:
            if os.path.exists(param_file):
                return param_file
        return None

    def _load_reduction_params(self, prop_name: str, model_info: Dict):
        """加载降维器参数（优先 final_models_dir，回退旧布局）"""
        safe_prop = prop_name.replace("（", "_").replace("）", "_").replace("/", "_")
        reduction_type = model_info.get('reduction_type', 'none')
        
        if reduction_type == 'none':
            return None
        
        param_file = self._find_param_file(f"{reduction_type}_params_{safe_prop}.pkl")
        if param_file is None:
            return None

        try:
            reducer = joblib.load(param_file)
            self.reduction_params[prop_name] = {
                'type': reduction_type,
                'reducer': reducer,
                'file': param_file
            }
            print(f"  加载{reduction_type.upper()}降维器参数: {param_file}")
            return reducer
        except Exception as e:
            print(f"  加载降维器参数失败: {e}")
        
        return None
    
    def _load_train_output_test_metrics(self) -> Dict:
        """从 results/property_prediction/training/output.json 读取测试集指标。

        预测必然发生在训练完成之后，training/output.json 的 test 部分是权威的
        泛化指标来源；当 best_model_info 未回写 test_r2/test_mape 时用它兜底。
        返回 {prop: {'r2': ..., 'mape': ...}}，找不到文件时返回 {}。
        """
        if self._train_test_metrics is not None:
            return self._train_test_metrics
        result = {}
        # 优先用当前工作目录，兜底用脚本所在目录向上找项目根
        candidates = [
            os.path.join(
                os.getcwd(), "results", "property_prediction", "training", "output.json"
            ),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "property_prediction", "training", "output.json"),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
                for prop, entry in data.items():
                    if isinstance(entry, dict) and isinstance(entry.get("test"), dict):
                        test = entry["test"]
                        r2 = test.get("r2")
                        mape = test.get("mape")
                        if isinstance(r2, (int, float)) and isinstance(mape, (int, float)):
                            result[prop] = {"r2": r2, "mape": mape}
                print(f"已从训练输出读取测试集指标: {path}（{len(result)} 个性能）")
                break
            except Exception as e:
                print(f"读取训练输出测试集指标失败: {path}: {e}")
                break
        self._train_test_metrics = result
        return result

    def _load_best_models_info(self):
        """加载所有性质的最佳模型信息"""
        print("=== 1. 加载最佳模型信息 ===")
        
        if not os.path.exists(self.best_models_dir):
            raise FileNotFoundError(f"最佳模型目录不存在: {self.best_models_dir}")
        
        # 查找所有最佳模型信息文件
        info_files = [f for f in os.listdir(self.best_models_dir) 
                     if f.startswith('best_model_info_') and f.endswith('.json')]
        
        if not info_files:
            raise FileNotFoundError(f"在 {self.best_models_dir} 中未找到最佳模型信息文件")
        
        loaded_properties = []
        
        for info_file in info_files:
            try:
                # 从文件名提取性质名称
                prop_name = info_file.replace('best_model_info_', '').replace('.json', '')
                
                # 如果指定了property_list，只加载列表中的性质
                if self.property_list is not None and prop_name not in self.property_list:
                    continue
                
                with open(os.path.join(self.best_models_dir, info_file), 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                
                self.best_models_info[prop_name] = model_info
                loaded_properties.append(prop_name)
                
                # 记录神经网络后端信息
                nn_backend = model_info.get('nn_backend', 'sklearn')
                self.nn_backend_info[prop_name] = nn_backend
                
                # 兜底：若 best_model_info 未回写 test 指标（旧模型文件），
                # 从 train_output/output.json 的 test 部分读取（预测必然发生在训练之后）
                if model_info.get('test_r2') is None or model_info.get('test_mape') is None:
                    train_test = self._load_train_output_test_metrics().get(prop_name, {})
                    if model_info.get('test_r2') is None and train_test.get('r2') is not None:
                        model_info['test_r2'] = train_test['r2']
                    if model_info.get('test_mape') is None and train_test.get('mape') is not None:
                        model_info['test_mape'] = train_test['mape']
                
                # 加载降维器参数
                self._load_reduction_params(prop_name, model_info)
                
                print(f"加载性质 '{prop_name}' 的最佳模型:")
                print(f"  - 模型类型: {model_info['best_model']}")
                print(f"  - 降维方式: {model_info.get('reduction_type', 'none').upper()}")
                print(f"  - 神经网络后端: {nn_backend}")
                
            except Exception as e:
                print(f"加载模型信息文件 {info_file} 失败: {e}")
                continue
        
        # 检查是否有指定的性质未找到
        if self.property_list is not None:
            missing_properties = [prop for prop in self.property_list if prop not in loaded_properties]
            if missing_properties:
                print(f"警告: 以下性质在最佳模型目录中未找到: {missing_properties}")
        
        print(f"成功加载 {len(self.best_models_info)} 个性质的最佳模型信息")
        return len(self.best_models_info) > 0
    
    def _load_models_and_features(self):
        """加载所有模型和特征列表"""
        print("\n=== 2. 加载模型和特征列表 ===")
        
        for prop_name, model_info in self.best_models_info.items():
            try:
                # 加载特征列表
                safe_prop = prop_name.replace("（", "_").replace("）", "_").replace("/", "_")
                features_file = os.path.join(self.best_models_dir, f'best_features_{safe_prop}.json')
                
                with open(features_file, 'r', encoding='utf-8') as f:
                    required_features = json.load(f)
                
                self.feature_lists[prop_name] = required_features
                
                # 加载模型（最终模型优先从 final_models_dir 读取，info/features 仍从 best_models_dir）
                nn_backend = self.nn_backend_info.get(prop_name, 'sklearn')
                
                if nn_backend == 'pytorch' and self.pytorch_available:
                    # 加载PyTorch模型
                    model_path = None
                    if self.final_models_dir:
                        cand = os.path.join(self.final_models_dir, f'best_model_{safe_prop}.pth')
                        if os.path.exists(cand):
                            model_path = cand
                    if model_path is None:
                        cand = os.path.join(self.best_models_dir, f'best_model_{safe_prop}.pth')
                        if os.path.exists(cand):
                            model_path = cand
                    if model_path is not None:
                        self.models[prop_name] = self._load_pytorch_model(model_path)
                        print(f"性质 '{prop_name}': PyTorch模型加载成功 ({model_path})")
                    else:
                        # print(f"警告: 性质 '{prop_name}' 的PyTorch模型文件不存在")
                        # 回退到sklearn
                        nn_backend = 'sklearn'
                
                if nn_backend == 'sklearn':
                    # 加载sklearn模型
                    model_path = None
                    if self.final_models_dir:
                        cand = os.path.join(self.final_models_dir, f'best_model_{safe_prop}.pkl')
                        if os.path.exists(cand):
                            model_path = cand
                    if model_path is None:
                        cand = os.path.join(self.best_models_dir, f'best_model_{safe_prop}.pkl')
                        if os.path.exists(cand):
                            model_path = cand
                    if model_path is not None:
                        self.models[prop_name] = joblib.load(model_path)
                        print(f"性质 '{prop_name}': sklearn模型加载成功 ({model_path})")
                    else:
                        # print(f"警告: 性质 '{prop_name}' 的模型文件不存在")
                        continue
                
            except Exception as e:
                print(f"加载性质 '{prop_name}' 的模型或特征列表失败: {e}")
                continue
        
        print(f"成功加载 {len(self.models)} 个模型和特征列表")
        return len(self.models) > 0
    
    def _load_pytorch_model(self, model_path: str):
        """加载PyTorch模型"""
        try:
            import torch
            from .pytorch_module import PyTorchModelSaver, PyTorchTrainer
            
            # 使用PyTorchModelSaver加载模型
            if PyTorchModelSaver:
                trainer = PyTorchModelSaver.load_model(model_path, device='cpu')
                return trainer
            else:
                raise ImportError("PyTorch模块不可用")
        except Exception as e:
            print(f"加载PyTorch模型失败: {e}")
            raise
    
    def _load_test_features(self):
        """加载测试集特征数据"""
        print("\n=== 3. 加载测试集特征数据 ===")
        
        try:
            if self.test_features_df is not None:
                loaded_sheets = []
                for prop_name, df in self.test_features_df.items():
                    if self.property_list is not None and prop_name not in self.property_list:
                        continue
                    self.test_data_dict[prop_name] = df
                    loaded_sheets.append(prop_name)
                    print(f"加载性质 '{prop_name}' 的测试数据: {df.shape}")
                
                if self.property_list is not None:
                    missing_properties = [prop for prop in self.property_list if prop not in loaded_sheets]
                    if missing_properties:
                        print(f"警告: 以下性质在传入的DataFrame中未找到: {missing_properties}")
                
                print(f"成功加载 {len(self.test_data_dict)} 个性质的测试数据（从DataFrame）")
                return len(self.test_data_dict) > 0
            
            raise ValueError("未提供测试特征数据 test_features_df")
            
        except Exception as e:
            print(f"加载测试集特征数据失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return False
    
    def _apply_reduction(self, prop_name: str, X_test: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
        """应用降维处理（修复：兼容已降维数据；未降维数据用保存的back_columns做变换并拼接前部分特征）"""
        if prop_name not in self.reduction_params:
            return X_test
        
        reduction_info = self.reduction_params[prop_name]
        reduction_type = reduction_info['type']
        info = reduction_info['reducer']  # 保存的降维信息dict（pca_info/svd_info/pls_info）
        
        print(f"  应用{reduction_type.upper()}降维处理...")
        
        # 降维成分列名前缀（SVD/tSVD 均命名为 SVD_主成分*）
        if reduction_type in ('svd', 'tsvd'):
            comp_prefix = 'SVD_主成分'
        else:
            comp_prefix = f'{reduction_type.upper()}_主成分'
        
        try:
            # 1) 数据已包含降维成分列（如SVD_主成分1）→ 上游已降维，直接按特征列表选取
            existing_reduced = [f for f in required_features if f.startswith(comp_prefix) and f in X_test.columns]
            if existing_reduced:
                available = [f for f in required_features if f in X_test.columns]
                missing = set(required_features) - set(available)
                if missing:
                    print(f"  警告: 已降维数据缺少特征: {sorted(missing)}，使用可用特征")
                else:
                    print(f"  数据已包含降维成分，直接选取特征: {len(available)}个")
                return X_test[available]
            
            # 2) 未降维：用保存的拟合模型对后部分特征(如MAT-*)做变换，再拼接前部分特征
            if isinstance(info, dict):
                fitted = (info.get(reduction_type) or info.get('svd') or info.get('pca')
                          or info.get('pls') or info.get('reducer'))
                front_columns = info.get('front_columns') or []
                back_columns = info.get('back_columns') or []
            else:
                # 兼容旧格式：info本身即为可transform对象
                fitted = info
                front_columns = [f for f in required_features if not f.startswith(comp_prefix)]
                back_columns = []
            
            if fitted is None or not hasattr(fitted, 'transform'):
                raise ValueError(f"降维信息中未找到可用的transform对象")
            
            available_back = [c for c in back_columns if c in X_test.columns]
            if not available_back:
                # 没有可用的后部分特征：直接使用前部分特征
                available_front = [c for c in front_columns if c in X_test.columns]
                print(f"  无可用后部分特征，直接使用前部分特征: {len(available_front)}个")
                return X_test[available_front]
            
            X_back = X_test[available_back].values.astype(np.float64)
            X_reduced = fitted.transform(X_back)
            
            # 成分列名：优先用保存的成分名，其次从required_features中取
            reduced_columns = list(info.get(f'{reduction_type}_columns') or [])
            if not reduced_columns:
                reduced_columns = [f for f in required_features if f.startswith(comp_prefix)]
            n_components = min(X_reduced.shape[1], len(reduced_columns))
            X_reduced_df = pd.DataFrame(X_reduced[:, :n_components],
                                        columns=reduced_columns[:n_components],
                                        index=X_test.index)
            
            # 前部分特征
            available_front = [c for c in front_columns if c in X_test.columns]
            X_front = X_test[available_front].reset_index(drop=True)
            X_reduced_df = X_reduced_df.reset_index(drop=True)
            final = pd.concat([X_front, X_reduced_df], axis=1)
            print(f"  降维完成: {X_test.shape} -> {final.shape}")
            return final
            
        except Exception as e:
            print(f"  降维处理失败: {e}")
            # 如果降维失败，尝试直接使用特征
            available_features = [f for f in required_features if f in X_test.columns]
            if available_features:
                print(f"  使用可用特征: {len(available_features)}个")
                return X_test[available_features]
            else:
                raise
    
    def _make_predictions(self):
        """进行预测"""
        print("\n=== 5. 进行预测 ===")
        
        total_predictions = 0
        
        # 如果指定了property_list，只预测列表中的性质
        properties_to_predict = list(self.test_data_dict.keys())
        print(f"将预测以下性质: {properties_to_predict}")
        
        for prop_name in properties_to_predict:
            if prop_name not in self.models:
                print(f"警告: 性质 '{prop_name}' 没有对应的模型，跳过预测")
                continue
            
            if prop_name not in self.feature_lists:
                print(f"警告: 性质 '{prop_name}' 没有特征列表，跳过预测")
                continue
            
            try:
                # 获取特征列表
                required_features = self.feature_lists[prop_name]
                
                # 获取测试数据
                X_test_df = self.test_data_dict[prop_name]
                
                # 应用降维处理（如果需要）
                if prop_name in self.reduction_params:
                    X_test_processed = self._apply_reduction(prop_name, X_test_df, required_features)
                else:
                    # 检查测试数据是否包含所有需要的特征
                    missing_features = set(required_features) - set(X_test_df.columns)
                    if missing_features:
                        print(f"警告: 性质 '{prop_name}' 的测试数据缺少以下特征: {list(missing_features)}")
                        # 只使用可用的特征
                        available_features = [f for f in required_features if f in X_test_df.columns]
                        if not available_features:
                            print(f"错误: 没有可用的特征，跳过预测")
                            continue
                        X_test_processed = X_test_df[available_features]
                    else:
                        X_test_processed = X_test_df[required_features]
                
                # 转换为numpy数组
                X_test = X_test_processed.values
                
                # 获取模型
                model = self.models[prop_name]
                
                # 进行预测
                print(f"性质 '{prop_name}': 开始预测 ({X_test.shape[0]} 个样本)...")
                print(f"  输入特征维度: {X_test.shape}")
                
                if isinstance(model, dict) or hasattr(model, 'predict'):
                    # sklearn模型或其他具有predict方法的模型
                    predictions = model.predict(X_test)
                elif hasattr(model, 'evaluate'):
                    # PyTorchTrainer
                    predictions = model.predict(X_test)
                else:
                    print(f"警告: 性质 '{prop_name}' 的模型类型不支持")
                    continue
                
                # 应用约束条件
                predictions = self._apply_constraints(prop_name, predictions)
                
                # 保存预测结果
                self.predictions_dict[prop_name] = predictions
                total_predictions += len(predictions)
                
                # 显示统计信息
                print(f"  预测完成: {len(predictions)} 个预测值")
                print(f"  预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
                print(f"  预测值均值: {predictions.mean():.4f}")
                
            except Exception as e:
                print(f"性质 '{prop_name}' 的预测失败: {e}")
                import traceback
                print(f"详细错误: {traceback.format_exc()}")
                continue
        
        print(f"预测完成，共生成 {total_predictions} 个预测值")
        return total_predictions > 0
    
    def _apply_constraints(self, prop_name: str, predictions: np.ndarray) -> np.ndarray:
        """应用约束条件"""
        constrained_predictions = predictions.copy()
        
        # 1. 应用非负约束
        if self.apply_non_negative:
            constrained_predictions = np.maximum(constrained_predictions, 0)
            print(f"  应用非负约束: 负值设为0")
        
        # 2. 应用范围约束
        if self.apply_range_constraint and prop_name in self.property_ranges:
            min_val, max_val = self.property_ranges[prop_name]
            original_count = len(constrained_predictions)
            constrained_predictions = np.clip(constrained_predictions, min_val, max_val)
            clipped_count = np.sum((constrained_predictions == min_val) | (constrained_predictions == max_val))
            if clipped_count > 0:
                print(f"  应用范围约束: {clipped_count}/{original_count} 个预测值被限制在 [{min_val:.4f}, {max_val:.4f}] 范围内")
        
        return constrained_predictions
    
    def _load_train_ranges(self):
        """加载训练集范围数据"""
        print("\n=== 4. 加载训练集范围数据 ===")
        
        if not self.apply_range_constraint:
            print("范围约束未启用，跳过训练集范围加载")
            return True
        
        try:
            if self.train_data_df is not None:
                train_df = self.train_data_df.copy()
            else:
                raise ValueError("未提供训练集范围数据 train_data_df")
            
            id_col_name = train_df.columns[0]
            train_df = train_df.set_index(id_col_name)
            
            properties_to_load = self.property_list if self.property_list is not None else self.best_models_info.keys()
            
            for prop_name in properties_to_load:
                if prop_name in train_df.columns:
                    prop_values = train_df[prop_name].dropna()
                    if len(prop_values) > 0:
                        min_val = prop_values.min()
                        max_val = prop_values.max()
                        self.property_ranges[prop_name] = (min_val, max_val)
                        print(f"性质 '{prop_name}': 训练集范围 = [{min_val:.4f}, {max_val:.4f}]")
                    else:
                        print(f"警告: 性质 '{prop_name}' 在训练集中没有有效数据")
                else:
                    print(f"警告: 性质 '{prop_name}' 不在训练集数据中")
            
            print(f"成功加载 {len(self.property_ranges)} 个性质的训练集范围")
            return True
            
        except Exception as e:
            print(f"加载训练集范围数据失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return False
    
    def _save_predictions(self):
        """保存预测结果到JSON"""
        print("\n=== 6. 保存预测结果到JSON ===")
        
        try:
            if not self.predictions_dict:
                raise ValueError("没有预测结果可保存")
            
            first_prop = list(self.test_data_dict.keys())[0]
            base_df = self.test_data_dict[first_prop].copy()
            
            result_df = pd.DataFrame()
            if '样本编号' in base_df.columns:
                result_df['样本编号'] = base_df['样本编号'].astype(str)
            else:
                result_df['样本编号'] = base_df.index.astype(str)
            result_df['数据集类型'] = '预测集' if self.predict else '测试集'
            
            for prop_name, predictions in self.predictions_dict.items():
                if len(predictions) != len(base_df):
                    raise ValueError(
                        f"目标 '{prop_name}' 的预测结果长度 ({len(predictions)}) "
                        f"与输入样本数 ({len(base_df)}) 不匹配"
                    )
                result_df[prop_name] = predictions
            
            meta_cols = ['样本编号', '数据集类型']
            pred_cols = [col for col in result_df.columns if col not in meta_cols]
            result_df = result_df[meta_cols + pred_cols]
            
            self.prediction_result_df = result_df

            prediction_records = {}
            for idx, row in self.prediction_result_df.iterrows():
                sample_id = str(row['样本编号'])
                prediction_records[sample_id] = {
                    col: float(row[col]) if pd.notna(row[col]) else None
                    for col in pred_cols
                }

            json_output_path = self.json_output_path if self.json_output_path else os.path.splitext(self.output_path)[0] + '.json'
            
            json_data = {
                'metadata': {
                    '生成时间': pd.Timestamp.now().isoformat(),
                    '预测模式': '新数据预测' if self.predict else '测试模式',
                    '目标数量': len(self.predictions_dict),
                    '样本数量': len(self.prediction_result_df),
                    '约束条件': {
                        'apply_non_negative': self.apply_non_negative,
                        'apply_range_constraint': self.apply_range_constraint
                    },
                    '使用的目标': list(self.predictions_dict.keys())
                },
                'predictions': {
                    'summary': {
                        prop_name: {
                            '预测样本数': len(predictions),
                            '预测最小值': float(np.min(predictions)),
                            '预测最大值': float(np.max(predictions)),
                            '预测平均值': float(np.mean(predictions)),
                            '预测标准差': float(np.std(predictions))
                        }
                        for prop_name, predictions in self.predictions_dict.items()
                    },
                    'dataset': prediction_records
                },
                'model_info': {
                    prop_name: {
                        'best_model': model_info.get('best_model', '未知'),
                        'reduction_type': model_info.get('reduction_type', 'none'),
                        'cv_avg_r2': model_info.get('cv_avg_r2', 'N/A'),
                        'cv_avg_mape': model_info.get('cv_avg_mape', 'N/A'),
                        'test_r2': model_info.get('test_r2', 'N/A'),
                        'test_mape': model_info.get('test_mape', 'N/A'),
                        '拟合状态': self._compute_fit_status(model_info)
                    }
                    for prop_name, model_info in self.best_models_info.items() 
                    if prop_name in self.predictions_dict
                }
            }
            
            if self.apply_range_constraint and self.property_ranges:
                json_data['constraints'] = {
                    'property_ranges': {
                        prop_name: {'min': float(min_val), 'max': float(max_val)}
                        for prop_name, (min_val, max_val) in self.property_ranges.items()
                    }
                }
            
            prediction_summary = self._build_prediction_summary()
            if prediction_summary:
                json_data['prediction_summary'] = prediction_summary
            
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"预测结果已保存到JSON: {json_output_path}")
            print(f"预测了 {len(self.predictions_dict)} 个性质")
            print(f"预测样本数: {len(result_df)}")
            print(f"输出格式: {'预测模式' if self.predict else '测试模式'}")
            
            return True
            
        except Exception as e:
            print(f"保存预测结果失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return False
    
    @staticmethod
    def _compute_fit_status(model_info: Dict) -> str:
        """根据训练集与测试集 R² 判定模型拟合状态（与 merge.py 口径一致）。

        1. 任一 R² 缺失          → 未知
        2. 训练集 R² < 0.5      → 欠拟合（连训练数据都未充分学习）
        3. train_r2 - test_r2 > 0.15 → 过拟合（训练好但泛化差）
        4. 其余                  → 正常拟合
        """
        train_r2 = model_info.get("train_eval_metrics", {}).get("r2")
        test_r2 = model_info.get("test_r2")  # 最终测试集评估 R²（评估阶段写回）
        if test_r2 is None:
            test_r2 = model_info.get("cv_avg_r2")  # 兜底：特征选择阶段的交叉验证 R²
        if train_r2 is None or test_r2 is None:
            return "未知"
        if not isinstance(train_r2, (int, float)) or not isinstance(test_r2, (int, float)):
            return "未知"
        if train_r2 < 0.5:
            return "欠拟合"
        if (train_r2 - test_r2) > 0.15:
            return "过拟合"
        return "正常拟合"

    def _build_prediction_summary(self):
        """构建预测汇总数据"""
        summary_data = []
        
        for prop_name, model_info in self.best_models_info.items():
            if prop_name in self.predictions_dict:
                predictions = self.predictions_dict[prop_name]
                train_r2 = model_info.get("train_eval_metrics", {}).get("r2", "N/A")
                test_score = model_info.get('test_score', 'N/A')
                reduction_type = model_info.get('reduction_type', 'none')
                nn_backend = model_info.get('nn_backend', 'sklearn')
                reduction_ratio = model_info.get('reduction_ratio', 'N/A')
                
                summary_info = {
                    '性质名称': prop_name,
                    '最佳模型': model_info.get('best_model', '未知'),
                    '神经网络后端': nn_backend,
                    '降维方式': reduction_type.upper(),
                    '降维比例': reduction_ratio,
                    '预测样本数': len(predictions),
                    '预测最小值': float(np.min(predictions)) if len(predictions) > 0 else 'N/A',
                    '预测最大值': float(np.max(predictions)) if len(predictions) > 0 else 'N/A',
                    '预测平均值': float(np.mean(predictions)) if len(predictions) > 0 else 'N/A',
                    '训练集R²': train_r2 if isinstance(train_r2, (int, float)) else 'N/A',
                    'test_score': test_score if isinstance(test_score, (int, float)) else 'N/A',
                    '约束条件': f"非负:{self.apply_non_negative}, 范围:{self.apply_range_constraint}"
                }
                
                if self.apply_range_constraint and prop_name in self.property_ranges:
                    min_val, max_val = self.property_ranges[prop_name]
                    clipped_count = np.sum((predictions == min_val) | (predictions == max_val))
                    total_count = len(predictions)
                    summary_info['范围约束'] = f"[{min_val:.4f}, {max_val:.4f}]"
                    summary_info['被约束样本数'] = f"{clipped_count}/{total_count}"
                
                summary_data.append(summary_info)
        
        if summary_data:
            print("预测汇总数据构建完成")
        
        return summary_data
    
    def run_pipeline(self, property_list: Optional[List[str]] = None):
        """运行完整预测流程"""
        print("="*60)
        print(f"开始多性质预测流程 - 模式: {'预测' if self.predict else '测试'}")
        print("="*60)
        
        if property_list is not None:
            self.set_property_list(property_list)
        
        steps = [
            self._load_best_models_info,
            self._load_models_and_features,
            self._load_test_features,
            self._load_train_ranges,
            self._make_predictions,
            self._save_predictions
        ]
        
        for i, step in enumerate(steps, 1):
            print(f"\n>>> 执行第{i}步: {step.__name__}")
            success = step()
            if not success:
                print(f"!!! 第{i}步执行失败，终止流程")
                return False
            print(f"<<< 第{i}步执行成功")
        
        print("\n" + "="*60)
        print("多性质预测完成")
        print("="*60)
        
        json_output_path = os.path.splitext(self.output_path)[0] + '.json'
        if os.path.exists(json_output_path):
            try:
                with open(json_output_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                print(f"输出文件 '{json_output_path}' 内容:")
                print(f"- 生成时间: {json_data['metadata']['生成时间']}")
                print(f"- 预测模式: {json_data['metadata']['预测模式']}")
                print(f"- 预测目标: {len(json_data['metadata']['使用的目标'])} 个")
                print(f"- 预测样本: {json_data['metadata']['样本数量']} 个")
                if self.prediction_result_df is not None:
                    print(f"- 数据形状: {self.prediction_result_df.shape}")
                    print(f"- 前5行预览:")
                    print(self.prediction_result_df.head())
            except Exception as e:
                print(f"读取JSON文件失败: {e}")
        
        return True

    def get_predictions_for_property(self, property_name: str):
        """获取指定性质的预测结果"""
        if property_name in self.predictions_dict:
            return self.predictions_dict[property_name]
        else:
            print(f"警告: 性质 '{property_name}' 没有预测结果")
            return None
    
    def get_prediction_summary(self) -> pd.DataFrame:
        """获取预测结果汇总信息"""
        summary_data = []
        
        for prop_name, predictions in self.predictions_dict.items():
            summary_data.append({
                '性质名称': prop_name,
                '预测样本数': len(predictions),
                '预测最小值': np.min(predictions),
                '预测最大值': np.max(predictions),
                '预测平均值': np.mean(predictions),
                '预测标准差': np.std(predictions)
            })
        
        return pd.DataFrame(summary_data)
