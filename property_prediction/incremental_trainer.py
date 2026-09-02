# incremental_trainer.py

import os
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import joblib
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback
from scipy.stats import pearsonr  # 新增：用于计算Pearson相关系数

from .model_optimizer_main import ModelOptimizer
from .pca_module import PCAReducer
from .pls_module import PLSReducer
from .svd_module import SVDReducer
from .pytorch_module import PyTorchConfig, PyTorchModelSaver
from .data_splitter import split_data_per_property

warnings.filterwarnings('ignore')

class IncrementalTrainer:
    def __init__(self, orig_features: int = 0, reduction_features: int = 50):
        self.model_optimizer = ModelOptimizer()
        self.property_info_cache = {}  # 缓存性质信息
        self.orig_features = orig_features
        self.reduction_features = reduction_features
    
    def _calculate_additional_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """计算额外的性能指标"""
        metrics = {}
        
        try:
            # 1. 计算Pearson相关系数
            if len(y_true) > 1 and len(y_pred) > 1:
                pearson_corr, pearson_p_value = pearsonr(y_true.values, y_pred)
                metrics['pearson_corr'] = float(pearson_corr)
                metrics['pearson_p_value'] = float(pearson_p_value)
            else:
                metrics['pearson_corr'] = None
                metrics['pearson_p_value'] = None
        except Exception:
            metrics['pearson_corr'] = None
            metrics['pearson_p_value'] = None
        
        # 2. 计算MAPE（平均绝对百分比误差）
        try:
            # 避免除以0
            mask = y_true != 0
            if mask.any():
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = 0
            metrics['mape'] = float(mape)
        except Exception:
            metrics['mape'] = None
        
        # 3. 计算RMSE
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(rmse)
        except Exception:
            metrics['rmse'] = None
        
        # 4. 计算MAE
        try:
            mae = mean_absolute_error(y_true, y_pred)
            metrics['mae'] = float(mae)
        except Exception:
            metrics['mae'] = None
        
        # 5. 计算R²
        try:
            r2 = r2_score(y_true, y_pred)
            metrics['r2'] = float(r2)
        except Exception:
            metrics['r2'] = None
        
        return metrics
    
    def load_existing_model_info(self, model_dir: str, property_names: List[str]) -> Dict[str, Dict]:
        """
        加载已有模型信息
        """
        property_info = {}
        
        for prop in property_names:
            safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")
            info_file = os.path.join(model_dir, f"best_model_info_{safe_prop}.json")
            
            if not os.path.exists(info_file):
                print(f"警告: 性质 '{prop}' 的模型信息文件不存在: {info_file}")
                continue
                
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                self.orig_features = info.get('orig_features', self.orig_features)
                
                property_info[prop] = {
                    'model_type': info.get('best_model', 'unknown'),
                    'reduction_type': info.get('reduction_type', 'none'),
                    'nn_backend': info.get('nn_backend', 'sklearn'),
                    'orig_features': self.orig_features,
                    'reduction_features': self.reduction_features,
                    'test_score': info.get('test_score', 0),
                    'train_r2': info.get('train_r2', 0),
                    'features': info.get('features', []),
                    'hyperparameters': info.get('hyperparameters', {}),
                    'train_eval_metrics': info.get('train_eval_metrics', {}),
                    'y_train_true': info.get('y_train_true', []),
                    'y_train_pred': info.get('y_train_pred', []),
                }
                print(f"已加载性质 '{prop}' 的信息: "
                    f"模型={property_info[prop]['model_type']}, "
                    f"降维={property_info[prop]['reduction_type']}, "
                    f"后端={property_info[prop]['nn_backend']}")
                
                # 打印原有额外指标（保留原逻辑）
                if 'train_eval_metrics' in property_info[prop] and property_info[prop]['train_eval_metrics']:
                    print(f"  原有额外指标:")
                    for metric_name, metric_value in property_info[prop]['train_eval_metrics'].items():
                        if metric_value is not None:
                            print(f"    {metric_name}: {metric_value:.4f}")
            
            except Exception as e:
                print(f"加载性质 '{prop}' 的模型信息失败: {e}")
                continue
        
        self.property_info_cache = property_info
        return property_info
    
    def setup_for_property(self, prop: str, property_info: Dict, judge: int = 1,
                           max_no_improve_rounds: int = 0) -> ModelOptimizer:
        """
        为特定目标设置优化器。
        """
        info = property_info[prop]
        
        # 创建新的优化器实例
        optimizer = ModelOptimizer()
        optimizer.property_names = [prop]
        optimizer.judge = judge
        optimizer.max_no_improve_rounds = max_no_improve_rounds
        
        optimizer.set_neural_network_backend(info['nn_backend'])
        reduction_type = info['reduction_type']
        reduction_features = info.get('reduction_features', 50)
        orig_features = info.get('orig_features', 0)
        
        if reduction_type == 'pca' and PCAReducer:
            optimizer.reducer = PCAReducer(reduction_features)
        elif reduction_type == 'pls' and PLSReducer:
            optimizer.reducer = PLSReducer(reduction_features)
        elif reduction_type == 'svd' and SVDReducer:
            optimizer.reducer = SVDReducer(reduction_features, svd_type='svd')
        elif reduction_type == 'tsvd' and SVDReducer:
            optimizer.reducer = SVDReducer(reduction_features, svd_type='tsvd')
        else:
            optimizer.reducer = None
        
        optimizer.reduction_features = reduction_features
        optimizer.orig_features = orig_features
        
        return optimizer
    
    def incremental_train_single_property(
        self,
        model_dir: str,
        training_params_dir: str,
        output_dir: str,
        prop: str,
        property_info: Dict,
        n_iterations: int = 10,
        n_folds: int = 5,
        search_method: str = 'random',
        judge: int = 1,
        max_no_improve_rounds: int = 0,
        additional_save_path: Optional[str] = None,
        train_features_df = None,
        train_targets_df = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        增量训练单个性质（逐性质独立划分模式）
        
        Args:
            judge: 特征选择模式 (0=复用已有模型特征, 1=启用贪心降维, 2=基于特征重要性的递归剔除)
            max_no_improve_rounds: judge=2时,test因子未提升后允许继续剔除的轮数上限
            additional_save_path: 额外的保存路径（可选）
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练结果字典
        """
        print(f"\n{'='*80}")
        print(f"开始增量训练性质: {prop}")
        judge_desc = {0: '复用已有模型特征', 1: '启用贪心降维', 2: '基于特征重要性的递归剔除'}.get(judge, '未知模式')
        print(f"judge参数: {judge} ({judge_desc})")
        if max_no_improve_rounds > 0:
            print(f"max_no_improve_rounds: {max_no_improve_rounds} (judge=2无提升容忍轮数)")
        if additional_save_path:
            print(f"额外保存路径: {additional_save_path}")
        print(f"逐性质划分: test_size={test_size}, random_state={random_state}")
        print(f"{'='*80}")
        
        info = property_info[prop]
        original_score = info.get('test_score', 0)
        original_r2 = info.get('train_r2', 0)
        
        # 设置优化器
        optimizer = self.setup_for_property(prop, property_info, judge, max_no_improve_rounds)
        optimizer.best_model_dir = model_dir
        optimizer.set_search_method(search_method)
        
        # 初始化模型配置（使用固定的模型类型）
        model_type = info['model_type']
        
        # 读取训练数据（逐性质独立划分）
        try:
            if train_features_df is not None and train_targets_df is not None:
                df_features_full = train_features_df.copy()
                df_targets_full = train_targets_df.copy()
            else:
                raise ValueError("必须提供 train_features_df 和 train_targets_df")

            # 逐性质独立划分训练/测试集 + 独立标准化
            X_train_scaled, y_train, X_test_scaled, y_test = split_data_per_property(
                df_features=df_features_full,
                df_targets=df_targets_full,
                property_name=prop,
                standardization_params_dir=training_params_dir,
                test_size=test_size,
                random_state=random_state,
                standardize=True,
            )

            # 存储测试集数据供后续评估
            id_col = df_features_full.columns[0]
            if len(X_test_scaled) > 0:
                test_data = {
                    'test_ids': X_test_scaled[id_col].tolist(),
                    'X_test_scaled': X_test_scaled,
                    'y_test': y_test,
                }
            else:
                test_data = {
                    'test_ids': [],
                    'X_test_scaled': X_test_scaled,
                    'y_test': y_test,
                }

            # 去掉ID列，转为优化器需要的格式
            X = X_train_scaled.set_index(id_col)
            y = y_train.copy()

            print(f"训练数据形状: X_train={X.shape}, y_train={y.shape}")

        except Exception as e:
            print(f"读取训练数据失败: {e}")
            return {'error': str(e)}
        
        # 如果是none模式且orig_features>0，仅使用前orig_features列特征
        if info['reduction_type'] == 'none' and info['orig_features'] > 0:
            actual_features = min(info['orig_features'], X.shape[1])
            X = X.iloc[:, :actual_features]
            print(f"使用'none'模式且orig_features={info['orig_features']}>0，仅使用前{actual_features}列特征")
        
        # judge=0时复用full模式筛选的完整前部分特征，后部分特征交由reducer按保存的降维类型/比例处理
        if judge == 0 and info.get('features'):
            reduced_prefixes = ('PCA_', 'PLS_', 'SVD_', 'TSVD_')
            front_cols = [f for f in info['features'] if not f.startswith(reduced_prefixes)]
            existing_front = [f for f in front_cols if f in X.columns]
            if existing_front:
                # 后部分特征按通用分组规则原样保留参与后续降维
                if optimizer.reducer:
                    _, back_candidates = optimizer.reducer.split_features(X)
                    back_cols = [c for c in back_candidates if c in X.columns]
                else:
                    back_cols = []
                X = X[existing_front + back_cols]
                print(f"judge=0: 复用full筛选的前部分特征 {len(existing_front)} 个，"
                      f"后部分特征 {len(back_cols)} 个交由{info['reduction_type'] or 'none'}降维处理")
            else:
                print(f"警告: 已有模型的前部分特征在新数据中不存在，退化为全量原始特征")
        
        # 只使用原来的模型类型进行训练
        model_list = [model_type]
        
        # 初始化模型配置
        optimizer.initialize_model_configs(
            reduction_type=info['reduction_type'],
            model_list=model_list
        )

        # judge=0: 固定使用full模型保存的降维比例，不参与超参搜索
        if judge == 0 and info.get('reduction_ratio') is not None:
            fixed_ratio = [float(info['reduction_ratio'])]
            for _cfg in optimizer.model_configs.values():
                if 'reduction_ratio' in _cfg['param_space']:
                    _cfg['param_space']['reduction_ratio'] = fixed_ratio
            print(f"judge=0: 固定降维比例 reduction_ratio={fixed_ratio[0]}")
        
        # 运行优化
        results, prop_results, best_model_info = optimizer.optimize_all_models(
            output_dir,  # output_dir
            training_params_dir,  # training_params_dir
            X, y, prop, model_list, n_iterations, n_folds,
            additional_save_path=additional_save_path  # 传递额外保存路径
        )
        
        if not best_model_info:
            print(f"性质 '{prop}' 训练失败，未找到最佳模型")
            return {'error': '训练失败'}
        
        # 计算额外指标
        additional_metrics = {}
        y_train_pred = None
        
        # 尝试从结果中获取预测值
        if results and model_type in results:
            if 'y_train_pred' in results[model_type]:
                y_train_pred = results[model_type]['y_train_pred']
                y_train_true = y
                
                # 计算额外指标
                additional_metrics = self._calculate_additional_metrics(y_train_true, y_train_pred)
                
                # # 将实际值和预测值添加到模型信息中
                # best_model_info['y_train_true'] = y_train_true.tolist()
                # best_model_info['y_train_pred'] = y_train_pred.tolist()
        
        # 将额外指标添加到模型信息中
        best_model_info['train_eval_metrics'] = additional_metrics

        # 保存训练集预测记录到 best_model_perform.json（与全量训练路径同格式，供可视化读取）
        if y_train_pred is not None:
            self._save_train_predictions(output_dir, prop, y_train_true, y_train_pred, additional_metrics)
        
        # 计算提升情况
        new_score = best_model_info.get('test_score', 0)
        new_r2 = best_model_info.get('train_r2', 0)
        
        score_improvement = new_score - original_score
        r2_improvement = new_r2 - original_r2
        
        # 在模型信息中添加judge参数
        best_model_info['judge'] = judge
        
        # 覆盖相关文件（传递额外保存路径）
        self.overwrite_model_files(model_dir, prop, best_model_info, additional_save_path)
        
        # 覆盖降维相关输出文件
        self.overwrite_reduction_files(output_dir, training_params_dir, prop, info['reduction_type'])
        
        # 打印额外指标
        if additional_metrics:
            print(f"\n增量训练额外指标 - {prop}:")
            for metric_name, metric_value in additional_metrics.items():
                if metric_value is not None:
                    print(f"  {metric_name}: {metric_value:.4f}")
        
        print(f"\n{'='*80}")
        print(f"增量训练完成 - {prop}")
        print(f"{'='*80}")
        print(f"原test因子: {original_score:.4f}, 新test因子: {new_score:.4f}, 提升: {score_improvement:.4f}")
        print(f"原训练R2: {original_r2:.4f}, 新训练R2: {new_r2:.4f}, 提升: {r2_improvement:.4f}")
        
        return {
            'property': prop,
            'model_type': model_type,
            'reduction_type': info['reduction_type'],
            'judge': judge,
            'original_score': original_score,
            'new_score': new_score,
            'score_improvement': score_improvement,
            'original_r2': original_r2,
            'new_r2': new_r2,
            'r2_improvement': r2_improvement,
            'train_eval_metrics': additional_metrics,  # 新增：包含额外指标
            'best_model_info': best_model_info,
            'test_data': test_data,  # 逐性质测试数据
        }

    def _save_train_predictions(self, output_dir: str, prop: str, y_true, y_pred,
                                additional_metrics: Dict = None):
        """保存训练集预测记录到 best_model_perform.json（与全量训练路径同格式）。

        供后续项点更新/统一完整评估的可视化读取：未重训项点仍能显示训练集散点/条形数据。
        """
        def _serialize(val):
            if isinstance(val, (np.integer,)):
                return int(val)
            if isinstance(val, (np.floating,)):
                return None if (np.isnan(val) or np.isinf(val)) else float(val)
            if isinstance(val, np.ndarray):
                return [_serialize(v) for v in val.tolist()]
            if isinstance(val, (list, tuple)):
                return [_serialize(v) for v in val]
            return val

        safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")

        if isinstance(y_true, pd.DataFrame):
            y_true_vals = y_true.iloc[:, 0].values
        elif hasattr(y_true, 'values'):
            y_true_vals = y_true.values
        else:
            y_true_vals = y_true

        predictions = []
        for exp_val, pred_val in zip(y_true_vals, y_pred):
            predictions.append({
                f'{prop}_exp': _serialize(exp_val),
                f'{prop}_pred': _serialize(pred_val),
            })

        metrics = {}
        if additional_metrics:
            metrics = {
                'R²': _serialize(additional_metrics.get('r2')),
                'RMSE': _serialize(additional_metrics.get('rmse')),
                'MAE': _serialize(additional_metrics.get('mae')),
                '平均相对误差(%)': None,
                'Pearson相关系数': _serialize(additional_metrics.get('pearson_corr')),
                'MAPE(%)': _serialize(additional_metrics.get('mape')),
                'Pearson p值': _serialize(additional_metrics.get('pearson_p_value')),
            }

        perform_file = os.path.join(output_dir, "best_model_perform.json")
        try:
            existing_data = {}
            if os.path.exists(perform_file):
                with open(perform_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            existing_data[safe_prop] = {
                'property_name': prop,
                'predictions': predictions,
                'metrics': metrics,
            }
            with open(perform_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            print(f"训练集预测记录已保存到: {perform_file} (性质: {prop}, 样本数: {len(predictions)})")
        except Exception as e:
            print(f"保存训练集预测记录失败: {e}")
    
    def overwrite_model_files(self, model_dir: str, prop: str, best_model_info: Dict, 
                            additional_save_path: Optional[str] = None):
        """覆盖模型相关文件
        
        Args:
            model_dir: 原始模型目录
            prop: 性质名称
            best_model_info: 最佳模型信息
            additional_save_path: 额外的保存路径（可选）
        """
        safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")
        
        # 1. 更新best_model_info.json（原始路径）
        info_file = os.path.join(model_dir, f"best_model_info_{safe_prop}.json")
        try:
            # 清理参数以便JSON保存
            cleaned_info = self.clean_params_for_json(best_model_info)
            
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_info, f, ensure_ascii=False, indent=2)
            print(f"已更新模型信息文件: {info_file}")
        except Exception as e:
            print(f"更新模型信息文件失败: {e}")
        
        # 2. 更新best_features.json（原始路径）
        features_file = os.path.join(model_dir, f"best_features_{safe_prop}.json")
        try:
            features = best_model_info.get('features', [])
            with open(features_file, 'w', encoding='utf-8') as f:
                json.dump(features, f, ensure_ascii=False, indent=2)
            print(f"已更新特征文件: {features_file}")
        except Exception as e:
            print(f"更新特征文件失败: {e}")
        
        # 3. 更新best_params.json（原始路径）
        params_file = os.path.join(model_dir, f"best_params_{safe_prop}.json")
        try:
            params = best_model_info.get('hyperparameters', {})
            cleaned_params = self.clean_params_for_json(params)
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_params, f, ensure_ascii=False, indent=2)
            print(f"已更新参数文件: {params_file}")
        except Exception as e:
            print(f"更新参数文件失败: {e}")
        
        # 4. 如果有PyTorch模型，复制.pth文件（原始路径）
        pth_file = os.path.join(model_dir, f"best_model_{safe_prop}.pth")
        if os.path.exists(pth_file):
            print(f"PyTorch模型文件已存在: {pth_file}")
        
        # 5. 复制模型.pkl文件（原始路径）
        pkl_file = os.path.join(model_dir, f"best_model_{safe_prop}.pkl")
        if os.path.exists(pkl_file):
            print(f"sklearn模型文件已存在: {pkl_file}")
        
        # 6. 如果有额外的保存路径，也保存一份
        if additional_save_path and os.path.exists(additional_save_path):
            try:
                # 创建额外的保存目录
                additional_model_dir = additional_save_path
                os.makedirs(additional_model_dir, exist_ok=True)
                
                # 复制所有模型文件到额外路径
                files_to_copy = [
                    f"best_model_info_{safe_prop}.json",
                    # f"best_features_{safe_prop}.json",
                    # f"best_params_{safe_prop}.json",
                    # f"best_model_{safe_prop}.pth",
                    # f"best_model_{safe_prop}.pkl"
                ]
                
                for file_name in files_to_copy:
                    source_path = os.path.join(model_dir, file_name)
                    dest_path = os.path.join(additional_model_dir, file_name)
                    
                    if os.path.exists(source_path):
                        shutil.copy2(source_path, dest_path)
                        print(f"  已复制 {file_name} 到额外路径: {additional_model_dir}")
                
                print(f"  模型文件已保存到额外路径: {additional_model_dir}")
                
            except Exception as e:
                print(f"  保存到额外路径失败: {e}")

        file_name = f"inc_{prop}.json"
        full_file_path = os.path.join(model_dir, file_name)

        with open(full_file_path, "w", encoding="utf-8") as f:
            json.dump({}, f)


    def overwrite_reduction_files(self, output_dir: str, training_params_dir: str, prop: str, reduction_type: str):
        """覆盖降维相关输出文件"""
        if reduction_type == 'none':
            return
        
        # 定义可能的降维文件
        reduction_files = [
            str(f"{output_dir}/{reduction_type}_summary.xlsx"),
            str(f"{training_params_dir}/{reduction_type}_params_{prop.replace('/', '_').replace('\\', '_')}.json")
        ]
        
        for file_path in reduction_files:
            if os.path.exists(file_path):
                try:
                    # 尝试移动文件（覆盖）
                    if file_path.startswith(str(training_params_dir)):
                        # 这是降维参数文件，应该已经存在
                        print(f"降维参数文件已更新: {file_path}")
                    elif file_path.startswith(str(output_dir)):
                        # 这是summary文件，原位保留
                        print(f"降维summary文件已生成: {file_path}")
                except Exception as e:
                    print(f"处理降维文件 {file_path} 失败: {e}")
    
    def clean_params_for_json(self, params: Dict) -> Dict:
        """清理参数以便JSON保存"""
        if params is None:
            return {}
        
        def convert_value(v):
            if v is None or isinstance(v, (bool, int, float, str)):
                return v
            
            try:
                import numpy as np
                if isinstance(v, np.integer):
                    return int(v)
                elif isinstance(v, np.floating):
                    return float(v)
                elif isinstance(v, np.ndarray):
                    return v.tolist()
                elif isinstance(v, np.bool_):
                    return bool(v)
            except:
                pass
            
            if isinstance(v, (list, tuple)):
                return [convert_value(item) for item in v]
            
            if isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            
            try:
                return str(v)
            except:
                return f"<无法序列化的对象: {type(v)}>"
        
        return convert_value(params)
    
    def backup_original_models(self, model_dir: str, backup_dir: str, property_names: List[str]):
        """备份原始模型文件"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{backup_dir}/backup_{timestamp}"
        
        os.makedirs(backup_path, exist_ok=True)
        
        for prop in property_names:
            safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")
            pattern = f"best_*_{safe_prop}.*"
            
            import glob
            files_to_backup = glob.glob(os.path.join(model_dir, pattern))
            
            for file_path in files_to_backup:
                try:
                    shutil.copy2(file_path, backup_path)
                    print(f"已备份: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"备份文件 {file_path} 失败: {e}")
        
        print(f"\n原始模型已备份到: {backup_path}")
        return backup_path
    
    def update_performance_file(self, output_dir: str, prop: str, original_score: float, 
                              new_score: float, score_improvement: float,
                              original_r2: float, new_r2: float, r2_improvement: float,
                              additional_metrics: Dict = None):
        """更新性能文件（包含额外指标），保存为JSON"""
        perform_file = os.path.join(output_dir, "incremental_train_perform.json")
        sheet_name = prop.replace("（", "_").replace("）", "_").replace("/", "_")

        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]

        improvement_data = {
            '指标': ['原test因子', '新test因子', '提升值', '提升百分比(%)',
                   '原训练R²', '新训练R²', 'R²提升值', 'R²提升百分比(%)'],
            '数值': [original_score, new_score, score_improvement, 
                   (score_improvement / abs(original_score) * 100) if original_score != 0 else 0,
                   original_r2, new_r2, r2_improvement,
                   (r2_improvement / abs(original_r2) * 100) if original_r2 != 0 else 0],
            '描述': ['原始测试因子', '增量训练后测试因子', '测试因子提升值', '相对提升百分比',
                   '原始训练R²', '增量训练后训练R²', 'R²提升值', 'R²相对提升百分比']
        }

        if additional_metrics:
            for metric_name, metric_value in additional_metrics.items():
                if metric_value is not None:
                    if metric_name == 'pearson_corr':
                        improvement_data['指标'].append('Pearson相关系数')
                        improvement_data['数值'].append(metric_value)
                        improvement_data['描述'].append('Pearson相关系数')
                    elif metric_name == 'mape':
                        improvement_data['指标'].append('MAPE(%)')
                        improvement_data['数值'].append(metric_value)
                        improvement_data['描述'].append('平均绝对百分比误差')
                    elif metric_name == 'pearson_p_value' and metric_value is not None:
                        improvement_data['指标'].append('Pearson p值')
                        improvement_data['数值'].append(metric_value)
                        improvement_data['描述'].append('Pearson相关系数的显著性p值')
                    elif metric_name == 'rmse':
                        improvement_data['指标'].append('RMSE')
                        improvement_data['数值'].append(metric_value)
                        improvement_data['描述'].append('均方根误差')
                    elif metric_name == 'mae':
                        improvement_data['指标'].append('MAE')
                        improvement_data['数值'].append(metric_value)
                        improvement_data['描述'].append('平均绝对误差')

        def _safe_val(v):
            if isinstance(v, (int, float, np.number)):
                if np.isnan(v) or np.isinf(v):
                    return None
                return float(v)
            return v

        improvement_data['数值'] = [_safe_val(v) for v in improvement_data['数值']]

        try:
            existing_data = {}
            if os.path.exists(perform_file):
                try:
                    with open(perform_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except Exception:
                    existing_data = {}

            existing_data[sheet_name] = improvement_data

            with open(perform_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4, default=str)

            print(f"增量训练结果已保存到: {perform_file}")
        except Exception as e:
            print(f"保存增量训练结果失败: {e}")

def incremental_train_main(
    model_dir: str,
    training_params_dir: str,
    property_names: List[str],
    output_dir: str = None,
    n_iterations: int = 10,
    n_folds: int = 5,
    search_method: str = 'random',
    judge: int = 1,
    max_no_improve_rounds: int = 0,
    batch_mode: bool = True,
    backup_original: bool = False,
    orig_features: int = 0,
    reduction_features: int = 50,
    additional_save_path: Optional[str] = None,
    train_features_df = None,
    train_targets_df = None,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    增量训练主函数
    
    Args:
        model_dir: 模型目录（包含已有模型）
        training_params_dir: 降维参数目录
        property_names: 要训练的性质列表
        output_dir: 输出目录（默认为模型目录的父目录）
        n_iterations: 超参数搜索迭代次数
        n_folds: 交叉验证折数
        search_method: 搜索方法（random或bayesian）
        judge: 特征选择模式 (0=复用已有模型特征, 1=启用贪心降维, 2=基于特征重要性的递归剔除)
        max_no_improve_rounds: judge=2时,test因子未提升后允许继续剔除的轮数上限
        batch_mode: 是否批量处理多个性质
        backup_original: 是否备份原始模型
        additional_save_path: 额外的保存路径（可选）
        test_size: 逐性质划分的测试集比例
        random_state: 随机种子
        **kwargs: 其他参数
        
    Returns:
        训练结果字典 和 逐性质测试数据
    """
    # 创建增量训练器
    trainer = IncrementalTrainer(orig_features=orig_features, reduction_features=reduction_features)

    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(model_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(training_params_dir, exist_ok=True)
    
    perform_file = os.path.join(output_dir, "incremental_train_perform.json")
    if os.path.exists(perform_file):
        os.remove(perform_file)
        print(f"已清除旧文件: {perform_file}")
    
    for filename in os.listdir(model_dir):
        file_path = os.path.join(model_dir, filename)
        if os.path.isfile(file_path) and filename.startswith("inc_") and filename.endswith(".json"):
            os.remove(file_path)
            print(f"已删除文件：{file_path}")

    # 加载已有模型信息
    print(f"加载已有模型信息...")
    property_info = trainer.load_existing_model_info(model_dir, property_names)
    
    if not property_info:
        print("错误: 没有找到有效的模型信息")
        return {}, {}
    
    # 备份原始模型（如果需要）
    backup_path = None
    if backup_original:
        backup_dir = os.path.join(output_dir, "backups")
        backup_path = trainer.backup_original_models(model_dir, backup_dir, property_names)
    
    results = {}
    per_property_test_data = {}
    
    # 逐个性质进行增量训练
    for prop in property_names:
        if prop not in property_info:
            print(f"跳过性质 '{prop}'，没有找到模型信息")
            continue
        
        try:
            result = trainer.incremental_train_single_property(
                model_dir=model_dir,
                training_params_dir=training_params_dir,
                output_dir=output_dir,
                prop=prop,
                property_info=property_info,
                n_iterations=n_iterations,
                n_folds=n_folds,
                search_method=search_method,
                judge=judge,
                max_no_improve_rounds=max_no_improve_rounds,
                additional_save_path=additional_save_path,
                train_features_df=train_features_df,
                train_targets_df=train_targets_df,
                test_size=test_size,
                random_state=random_state,
            )
            
            if 'error' not in result:
                results[prop] = result
                # 收集逐性质测试数据
                if 'test_data' in result:
                    per_property_test_data[prop] = result['test_data']
                
                # 更新性能文件（包含额外指标）
                trainer.update_performance_file(
                    output_dir, prop,
                    result['original_score'], result['new_score'],
                    result['score_improvement'],
                    result['original_r2'], result['new_r2'],
                    result['r2_improvement'],
                    result.get('train_eval_metrics', {})  # 传递额外指标
                )
                
                print(f"\n 性质 '{prop}' 增量训练完成")
            else:
                print(f" 性质 '{prop}' 增量训练失败: {result['error']}")
                
        except Exception as e:
            print(f" 性质 '{prop}' 增量训练异常: {e}")
            traceback.print_exc()
    
    # 打印汇总结果
    if results:
        print(f"\n{'='*80}")
        print("增量训练汇总结果")
        print(f"{'='*80}")
        
        for prop, result in results.items():
            print(f"\n性质: {prop}")
            print(f"  模型类型: {result['model_type']}")
            print(f"  降维方式: {result['reduction_type']}")
            print(f"  特征降维: {'启用' if result['judge'] > 0 else '禁用'}")
            print(f"  原test因子: {result['original_score']:.4f}")
            print(f"  新test因子: {result['new_score']:.4f}")
            print(f"  提升值: {result['score_improvement']:.4f}")
            print(f"  提升百分比: {(result['score_improvement']/abs(result['original_score'])*100 if result['original_score']!=0 else 0):.2f}%")
            
            # 打印额外指标
            if 'train_eval_metrics' in result and result['train_eval_metrics']:
                print(f"  额外指标:")
                for metric_name, metric_value in result['train_eval_metrics'].items():
                    if metric_value is not None:
                        print(f"    {metric_name}: {metric_value:.4f}")
        
        print(f"\n 增量训练完成！")
        if backup_path:
            print(f"原始模型已备份到: {backup_path}")
        if additional_save_path:
            print(f"模型信息已保存到额外路径: {additional_save_path}")
    else:
        print(f"\n 没有成功完成任何性质的增量训练")
    
    return results, per_property_test_data

# Jupyter调用示例（更新）
if __name__ == "__main__":
    print("增量训练模块已更新为DataFrame直传模式，请通过Excute_pipe.py调用")
    pass
