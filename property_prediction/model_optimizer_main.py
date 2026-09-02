import random
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic, DotProduct
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import xgboost as xgb
import os
from itertools import combinations
import joblib 
import json
import pickle
import shutil
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
import catboost as cb
import lightgbm as lgb
from .pca_module import PCAReducer
from .pls_module import PLSReducer
from .svd_module import SVDReducer
from .pytorch_module import PyTorchConfig, PyTorchModelSaver
from sklearn.preprocessing import StandardScaler
from .bayesian_optimizer import BayesianHyperparamSearcher
from .data_splitter import split_data_per_property
from .feature_importance import compute_feature_importance
from .model_input import prepare_model_input
import torch
from pathlib import Path

warnings.filterwarnings('ignore')


class ModelOptimizer:
    def __init__(self):
        self.model_configs = {}
        self.judge = 0
        self.max_no_improve_rounds = 0  # judge=2 递归剔除：test因子未提升后允许继续剔除的轮数上限
        self.property_names = []
        self.best_model_dir = "best_models"
        self.reducer = None  # 降维器实例
        self.svd_type = 'svd'
        self.enable_figures_base64 = True  # 是否生成base64图片（figures.json并合并到output.json），默认关闭；前端直接用chart_data.json数据文件生成图表

        # 神经网络配置
        self.nn_backend = 'sklearn'  # 默认使用sklearn
        self.nn_config = None  # PyTorch配置
        
        # PLUS模式相关
        self.plus_mode = False
        self.custom_mode = False  # 新增：custom模式标志
        self.mode_list = []  # 新增：存储自定义模式列表
        self.plus_results = {}  # 存储PLUS模式各子模式结果
        self.reduction_features = 50  # 降维特征数
        self.orig_features = 0  # 原始特征数量（新增参数）
        
        # 贝叶斯优化相关
        self.search_method = 'random'  # 默认使用随机搜索
        self.bayesian_searcher = None  # 贝叶斯搜索器
        
        # 模型性能数据（替代Excel存储）
        self._best_model_perform_data = {}

        # 逐性质划分参数
        self.test_size = 0.2
        self.random_state = 42
        self._per_property_test_data = {}  # {prop: (X_test_df, y_test_series)}

    def _print_device_info_(self):
        """打印设备及内存占用信息"""
        print("="*50)
        print("训练设备信息")
        print("="*50)
        # CUDA信息
        print(f"PyTorch: {torch.__version__} | CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                used = torch.cuda.memory_allocated(i)/1024**3
                total = props.total_memory/1024**3
                print(f"GPU {i}: {props.name} | 显存: {used:.2f}/{total:.2f} GB")
        else:
            print("")

    def _print_device_info(self):
        """打印设备信息"""
        print("=" * 50)
        print("训练设备信息")
        print("=" * 50)
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("GPU不可用，使用CPU训练")

    def set_search_method(self, method: str = 'random'):
        """设置超参数搜索方法"""
        valid_methods = ['random', 'bayesian']
        if method not in valid_methods:
            raise ValueError(f"无效的搜索方法: {method}，有效值为: {valid_methods}")
        
        self.search_method = method
        print(f"超参数搜索方法设置为: {method}")
        
        if method == 'bayesian':
            try:
                # 只创建搜索器，不在此时初始化优化器（因为model_configs可能还没准备好）
                self.bayesian_searcher = BayesianHyperparamSearcher(self)
                print("贝叶斯搜索器已创建")
            except Exception as e:
                print(f"初始化贝叶斯搜索器失败: {e}")
                print("将使用随机搜索")
                self.search_method = 'random'
                self.bayesian_searcher = None

    def clean_model_directory(self, output_dir, directory_path, target_properties, model_list):
        """
        清理模型目录中涉及目标性质的所有文件
        
        Args:
            directory_path (str): 模型目录路径
            target_properties (list): 目标性质列表
            model_list (list): 模型类型列表（此参数保留以兼容接口，实际未使用）
        """
        deleted_files = []
        kept_files = []

        if not os.path.exists(directory_path):
            print(f"    {directory_path} 目录不存在，无需清理")
            return deleted_files, kept_files        
        
        # 获取所有目标性质的安全名称（兼容特殊字符替换）
        safe_props = [prop.replace("（", "_").replace("）", "_").replace("/", "_")
                    for prop in target_properties]
        
        print(f"   正在清理目录: {directory_path}")
        print(f"   清理涉及的性质: {target_properties}")
        
        # 核心逻辑：遍历文件，删除包含任意目标性质名称的文件
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # 跳过目录（最后统一清理空目录）
            if os.path.isdir(file_path):
                continue
            
            # 判断文件是否涉及目标性质（文件名包含任意一个安全名称）
            is_target_file = any(safe_prop in filename for safe_prop in safe_props)
            
            if is_target_file:
                # 删除涉及目标性质的文件
                try:
                    os.remove(file_path)
                    deleted_files.append(filename)
                    print(f"      已删除: {filename}")
                except Exception as e:
                    print(f"      删除 {filename} 时出错: {e}")
            else:
                # 保留不相关文件
                kept_files.append(filename)
        
        # 清理空的子目录（基础清理）
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path):
                if not os.listdir(item_path):
                    try:
                        os.rmdir(item_path)
                        print(f"      删除空目录: {item}")
                    except Exception as e:
                        print(f"      删除空目录 {item} 时出错: {e}")
        
        # 输出统计
        if deleted_files:
            print(f"   已从 {directory_path} 中删除 {len(deleted_files)} 个涉及目标性质的文件")
            print(f"   保留 {len(kept_files)} 个无关文件")
        else:
            print(f"    {directory_path} 中没有需要删除的目标性质文件")
        
        return deleted_files, kept_files

    def set_neural_network_backend(self, backend: str = 'sklearn', config: Optional[Dict] = None):
        """设置神经网络后端"""
        valid_backends = ['sklearn', 'pytorch']
        if backend not in valid_backends:
            raise ValueError(f"无效的后端: {backend}，有效值为: {valid_backends}")
        
        self.nn_backend = backend
        if config:
            self.nn_config = config
            
        if backend == 'pytorch':
            print(f"PyTorch后端已启用")
            self._print_device_info()
            if config:
                print(f"PyTorch配置: {config}")
    
    def initialize_model_configs(self, tree_i=2, tree_j=6, line_i=2, line_j=10, 
                                 fnn_i=2, fnn_j=4, reduction_type='none',
                                 model_list=None):
        """初始化模型配置参数
        
        Args:
            model_list: 要初始化的模型列表，为None时初始化所有模型
        """
        # 基础配置
        base_configs = {
            'xgb': {
                'i': tree_i,
                'j': tree_j,
                'param_space': {
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'n_estimators': [50, 100, 200, 300],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2]
                }
            },
            'rf': {
                'i': tree_i,
                'j': tree_j,
                'param_space': {
                    'max_depth': [None, 5, 10, 15, 20],
                    'n_estimators': [50, 100, 200, 300],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                }
            },
            'gbr': {
                'i': tree_i,
                'j': tree_j,
                'param_space': {
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'hist_gbdt': {
                'i': tree_i,
                'j': tree_j,
                'param_space': {
                    'max_depth': [3, 5, 7, 9, None],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'max_iter': [50, 100, 200, 300],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'l2_regularization': [0.0, 0.1, 0.2, 0.5],
                    'max_bins': [88, 128, 255],
                    'early_stopping': [True, False]
                }
            },
            'extra_trees': {
                'i': tree_i,
                'j': tree_j,
                'param_space': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None, 0.5, 0.8],
                    'bootstrap': [True, False]
                }
            },
            'gbdt': {
                'i': tree_i,
                'j': tree_j,
                'param_space': {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'dt': {
                'i': line_i,
                'j': line_i,
                'param_space': {
                    'max_depth': [None, 3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'ridge': {
                'i': line_i,
                'j': line_j,
                'param_space': {
                    'alpha': [0.001, 0.01, 0.03, 0.05, 0.1],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
                    'max_iter': [1000, 2000]
                }
            },
            'lasso': {
                'i': line_i,
                'j': line_j,
                'param_space': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                    'max_iter': [7000,10000],
                    'tol': [0.001, 0.01]
                }
            },
            'elasticnet': {
                'i': line_i,
                'j': line_j,
                'param_space': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_iter': [5000, 7000, 10000],
                    'tol': [0.001, 0.01]
                }
            },
            'linear': {
                'i': line_i,
                'j': line_j,
                'param_space': {
                    'fit_intercept': [True, False],
                    'copy_X': [True, False],
                    'positive':[True, False]
                }
            },
            'svr': {
                'i': tree_i,
                'j': tree_j,
                'param_space': {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 5],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'epsilon': [0.01, 0.1, 0.2, 0.5]
                }
            },
            'svr_rbf': {
                'i': tree_i,
                'j': tree_j,
                'param_space': {
                    'C': [0.1, 1, 5],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5]
                }
            },
            'catboost': {
                'i': tree_i,
                'j': tree_j,
                'param_space': {
                    'n_estimators': [100, 200, 300, 500],
                    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
                    'depth': [4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'border_count': [32, 64, 128, 255],
                    'bagging_temperature': [0, 0.5, 1, 2],
                    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
                    'min_data_in_leaf': [1, 3, 5, 7]
                } 
            },
            'lgbm': {
                'i': tree_i,
                'j': tree_j,
                'param_space' : {
                    'num_leaves': [31, 50, 63],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 8],  # 增加浅树，减少过拟合风险
                    'min_child_samples': [1, 5, 10, 20],  # 调整范围，允许更小的节点
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'reg_alpha': [0, 0.001, 0.01, 0.1],
                    'reg_lambda': [0, 0.001, 0.01, 0.1],
                    'min_gain_to_split': [0, 0.001, 0.01],  # 调整范围，允许0增益分裂
                }
            },
            'adaboost': {
                'i': tree_i,
                'j': tree_j,
                'param_space': {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'loss': ['linear', 'square', 'exponential'],
                    'estimator__max_depth': [1, 2, 3, 5]  # 注意：这是嵌套参数
                }
            },
            'knn': {
                'i': line_i,
                'j': line_j,
                'param_space': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [20, 30, 40, 50],
                    'p': [1, 2],  # 距离度量，1为曼哈顿距离，2为欧氏距离
                    # 'metric': ['minkowski', 'euclidean', 'manhattan']
                }
            },
            'gpr': {
                'i': line_i,
                'j': line_j,
                'param_space': {
                    'kernel': [
                        # 1.0 * RBF(length_scale=1.0),
                        # 1.0 * Matern(length_scale=1.0, nu=1.5),
                        # 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
                        # C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
                        RBF(length_scale=1.0),  # 直接使用kernel对象
                        Matern(length_scale=1.0, nu=1.5),
                        RationalQuadratic(length_scale=1.0, alpha=0.1),
                        C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
                    ],
                    'alpha': [1e-10, 1e-8, 1e-6, 1e-4],
                    'n_restarts_optimizer': [0, 1, 2, 5],
                    'optimizer': ['fmin_l_bfgs_b', None],
                    'normalize_y': [True, False]
                }
            },
            'linearsvr': {
                'i': line_i,
                'j': line_j,
                'param_space': {
                    'epsilon': [0.0, 0.01, 0.1, 0.2],
                    'C': [0.1, 0.5, 1, 5, 10],
                    'loss': ['squared_epsilon_insensitive'],
                    'dual': [True, False],
                    'tol': [1e-4, 1e-3, 1e-2],
                    'max_iter': [1000, 2000, 5000]
                }
            },
            'huber': {
                'i': line_i,
                'j': line_j,
                'param_space': {
                    'epsilon': [1.1, 1.35, 1.5, 2.0],
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                    'max_iter': [100, 200, 500, 1000],
                    'tol': [1e-4, 1e-3, 1e-2],
                    'warm_start': [True, False]
                }
            },
            'poly': {
                'i': line_i,
                'j': line_j,
                'param_space': {
                    'degree': [1, 2, 3],
                    'interaction_only': [True, False],
                    'include_bias': [True, False],
                    'linear__fit_intercept': [True, False],
                }
            },
            'bayesian_ridge': {
                'i': line_i,
                'j': line_j,
                'param_space': {
                    'max_iter': [100, 200, 300, 500],
                    'tol': [1e-3, 1e-4, 1e-5],
                    'alpha_1': [1e-6, 1e-5, 1e-4],
                    'alpha_2': [1e-6, 1e-5, 1e-4],
                    'lambda_1': [1e-6, 1e-5, 1e-4],
                    'lambda_2': [1e-6, 1e-5, 1e-4],
                    'compute_score': [True, False],
                    'fit_intercept': [True, False],
                }
            }
        }
        
        # 根据后端选择神经网络配置
        if self.nn_backend == 'sklearn':
            # sklearn神经网络配置
            base_configs['fnn'] = {
                'i': fnn_i,
                'j': fnn_j,
                'param_space': {
                    'hidden_layer_sizes': [
                        (50,), (100,), (50, 50), (100, 50), 
                        (200,), (100, 100), (50, 25, 10), (100, 50, 25)
                    ],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam'],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'max_iter': [1000, 1500, 2000],
                    'early_stopping': [True, False],
                    'validation_fraction': [0.1, 0.2],
                    'n_iter_no_change': [10, 20, 50],
                    'tol': [1e-4, 1e-3],
                    'batch_size_ratio': [0.1, 0.2, 0.3, 0.5]
                }
            }
            
            base_configs['deep_fnn'] = {
                'i': fnn_i,
                'j': fnn_j,
                'param_space': {
                    'hidden_layer_sizes': [
                        (100, 50), (200, 100), (100, 100, 50), 
                        (200, 100, 50), (100, 50, 25, 10), (200, 100, 50, 25)
                    ],
                    'activation': ['relu'],
                    'solver': ['adam'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant','adaptive'],
                    'learning_rate_init': [0.001, 0.01],
                    'max_iter': [2500, 3500],
                    'early_stopping': [True],
                    'validation_fraction': [0.1, 0.2],
                    'n_iter_no_change': [20, 50],
                    'tol': [1e-4],
                    'batch_size_ratio': [0.1, 0.2, 0.3, 0.5]
                }
            }
            
            base_configs['simple_fnn'] = {
                'i': fnn_i,
                'j': fnn_j,
                'param_space': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam'],
                    'alpha': [0.001, 0.01],
                    'learning_rate': ['constant'],
                    'learning_rate_init': [0.01],
                    'max_iter': [500, 1000],
                    'early_stopping': [False],
                    'validation_fraction': [0.1],
                    'n_iter_no_change': [10],
                    'tol': [1e-3],
                    'batch_size_ratio': [0.1, 0.2, 0.3, 0.5]
                }
            }
            
            # 为sklearn后端添加resnet占位符
            # 注意：sklearn的MLPRegressor不是真正的残差网络，仅作为MLP近似
            # early_stopping=False避免小样本时验证集过小报错
            base_configs['resnet'] = {
                'i': fnn_i,
                'j': fnn_j,
                'param_space': {
                    'hidden_layer_sizes': [(50, 50), (100, 100), (200, 200), 
                                          (100, 100, 100), (150, 150, 150)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01],
                    'max_iter': [500, 1000, 2000],
                    'early_stopping': [False]
                }
            }
            
        elif self.nn_backend == 'pytorch':
            # PyTorch神经网络配置
            pytorch_configs = PyTorchConfig.get_model_configs(fnn_i, fnn_j)
            base_configs.update(pytorch_configs)
        
        # 如果使用降维，添加降维比例参数
        # 注：该固定19档比例仅供贝叶斯搜索使用；随机搜索独立按后部分特征量生成离散比例（_build_reduction_ratio_list），
        # 以避免与模型超参做笛卡尔积造成空间膨胀。
        if reduction_type in  ['pca', 'pls', 'svd', 'tsvd']:
            ratio_space = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                           0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
            for model_name, config in base_configs.items():
                config['param_space']['reduction_ratio'] = ratio_space
        
        if model_list is not None:
            base_configs = {k: v for k, v in base_configs.items() if k in model_list}
            print(f"仅初始化指定模型配置 ({len(base_configs)} 个): {list(base_configs.keys())}")
        
        self.model_configs = base_configs

        if self.search_method == 'bayesian':
            try:
                if self.bayesian_searcher is None:
                    print(f"初始化贝叶斯优化器...")
                    self.bayesian_searcher = BayesianHyperparamSearcher(self)
                    if self.bayesian_searcher.bayesian_optimizer is None:
                        print("警告: 贝叶斯优化器初始化失败，将使用随机搜索")
                        self.search_method = 'random'
                else:
                    # 如果已经存在，重新初始化优化器
                    print(f"重新初始化贝叶斯优化器...")
                    self.bayesian_searcher._initialize_bayesian_optimizers()
            except Exception as e:
                print(f"贝叶斯优化器初始化失败: {e}")
                print("将使用随机搜索")
                self.search_method = 'random'
                self.bayesian_searcher = None
    
    def create_model_with_params(self, model_type: str, params: Optional[Dict] = None, 
                               input_size: Optional[int] = None):
        """创建模型实例"""
        if params is None:
            params = {}
        
        if model_type not in self.model_configs:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 移除降维比例参数（不是模型参数）
        model_params = params.copy()
        model_params.pop('reduction_ratio', None)
        
        # 处理神经网络模型
        if model_type in ['fnn', 'deep_fnn', 'simple_fnn', 'resnet']:
            if self.nn_backend == 'sklearn':
                # sklearn后端
                # 小数据集强制禁用early_stopping，避免"validation set too small"错误
                model_params['early_stopping'] = False
                batch_size_ratio = model_params.pop('batch_size_ratio', None)
                if batch_size_ratio is not None:
                    batch_size = 'auto'  # sklearn会自动调整
                    model_params['batch_size'] = batch_size
                return MLPRegressor(**model_params, random_state=42)
                
            elif self.nn_backend == 'pytorch':
                # PyTorch后端
                if input_size is None:
                    raise ValueError(f"对于PyTorch模型 {model_type}，需要指定input_size参数")
                
                # 添加模型类型到参数中
                model_params['model_type'] = model_type
                
                # 创建PyTorch模型训练器
                trainer = PyTorchConfig.create_model(model_type, input_size, model_params)
                return trainer
        
        # 其他模型
        elif model_type == 'xgb':
            return xgb.XGBRegressor(**model_params, objective='reg:squarederror', random_state=42)
        elif model_type == 'rf':
            return RandomForestRegressor(**model_params, random_state=42)
        elif model_type == 'hist_gbdt':
            return HistGradientBoostingRegressor(**model_params, random_state=42)
        elif model_type == 'gbdt':
            return GradientBoostingRegressor(**model_params, random_state=42)
        elif model_type == 'extra_trees':
            return ExtraTreesRegressor(**model_params, random_state=42)
        elif model_type == 'gbr':
            return GradientBoostingRegressor(**model_params, random_state=42)
        elif model_type == 'dt':
            return DecisionTreeRegressor(**model_params, random_state=42)
        elif model_type == 'ridge':
            return Ridge(**model_params, random_state=42)
        elif model_type == 'lasso':
            return Lasso(**model_params, random_state=42)
        elif model_type == 'elasticnet':
            return ElasticNet(**model_params, random_state=42)
        elif model_type == 'linear':
            return LinearRegression(**model_params)
        elif model_type == 'svr':
            return SVR(**model_params)
        elif model_type == 'svr_rbf':
            model_params['kernel'] = 'rbf'
            return SVR(**model_params)
        
        # 新增模型
        elif model_type == 'catboost':
            return cb.CatBoostRegressor(**model_params, random_seed=42, verbose=0)
        
        elif model_type == 'lgbm':
            return lgb.LGBMRegressor(**model_params, random_state=42, verbosity=-1)
        
        elif model_type == 'adaboost':
            # 处理嵌套参数（estimator__max_depth）
            estimator_params = {}
            base_params = {}
            for key, value in model_params.items():
                if key.startswith('estimator__'):
                    estimator_key = key.replace('estimator__', '')
                    estimator_params[estimator_key] = value
                else:
                    base_params[key] = value
            
            # 创建基学习器
            if estimator_params:
                base_estimator = DecisionTreeRegressor(**estimator_params, random_state=42)
            else:
                base_estimator = DecisionTreeRegressor(max_depth=3, random_state=42)
            
            return AdaBoostRegressor(estimator=base_estimator, **base_params, random_state=42)
        
        elif model_type == 'knn':
            return KNeighborsRegressor(**model_params)
        
        elif model_type == 'gpr':
            return GaussianProcessRegressor(**model_params, random_state=42)
        
        elif model_type == 'linearsvr':
            return LinearSVR(**model_params, random_state=42)
        
        elif model_type == 'huber':
            return HuberRegressor(**model_params)
        
        elif model_type == 'poly':
            # 多项式回归需要使用Pipeline
            degree = model_params.pop('degree', 2)
            interaction_only = model_params.pop('interaction_only', False)
            include_bias = model_params.pop('include_bias', True)
            
            # 创建多项式特征转换器
            poly = PolynomialFeatures(
                degree=degree, 
                interaction_only=interaction_only, 
                include_bias=include_bias
            )
            
            # 处理线性回归参数
            linear_params = {}
            for key in list(model_params.keys()):
                if key.startswith('linear__'):
                    linear_key = key.replace('linear__', '')
                    linear_params[linear_key] = model_params.pop(key)
            
            # 创建线性回归
            linear = LinearRegression(**linear_params)
            
            # 创建Pipeline
            return Pipeline([
                ('poly', poly),
                ('linear', linear)
            ])
        
        elif model_type == 'bayesian_ridge':
            return BayesianRidge(**model_params)
        
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def calculate_test_score(self, X: pd.DataFrame, y: pd.Series, model_type: str, 
                           params: Optional[Dict] = None, n_folds: int = 5) -> Tuple[float, float, float, float, float]:
        """计算综合测试因子"""
        total_rmse_test = 0
        total_mae_test = 0
        total_r2_test = 0
        mean_property = y.mean()
        if not np.isfinite(mean_property) or mean_property == 0:
            mean_property = 1.0  # 防止NaN/Inf/0导致test_score异常
        total_mape_test = 0

        total_folds = 0
        positive_r2_count = 0
        max_r2 = -float('inf')
        
        if 5 < len(y) < 9:  # 长度大于5且小于10
            n_folds = 3
        elif len(y) <= 5:    # 长度小于等于5（即＜6）
            n_folds = 2

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=60)
        X.columns = X.columns.astype(str)
        
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            # 重置索引以确保对齐
            X_train = X_train.reset_index(drop=True)
            X_val = X_val.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_val = y_val.reset_index(drop=True)
            
            # scaler = StandardScaler()
            
            # # 训练集标准化
            # X_train_scaled = scaler.fit_transform(X_train)
            # X_train_scaled = pd.DataFrame(
            #     X_train_scaled, 
            #     columns=X_train.columns, 
            #     index=X_train.index
            # )
            
            # # 验证集使用训练集的标准化参数
            # X_val_scaled = scaler.transform(X_val)
            # X_val_scaled = pd.DataFrame(
            #     X_val_scaled, 
            #     columns=X_val.columns, 
            #     index=X_val.index
            # )
            
            # 交叉验证不依次标准化，训练途中标准化信息共享，减少数据质量波动
            X_train_scaled = X_train
            X_val_scaled = X_val

            # print(f"  第{fold_idx+1}折: 训练集标准化完成 (均值: {scaler.mean_[0]:.2f}, 方差: {scaler.var_[0]:.2f})")
            
            # 应用降维
            if self.reducer:
                reduction_ratio = params.get('reduction_ratio', 0.5) if params else 0.5
                
                # 使用标准化后的数据进行降维
                X_train_processed, X_val_processed, _ = self.reducer.reduce_fold(
                    X_train_scaled, X_val_scaled, y_train, ratio=reduction_ratio
                )
            else:
                X_train_processed, X_val_processed = X_train_scaled, X_val_scaled
            
            # 确保数据维度匹配
            if len(X_train_processed) != len(y_train):
                print(f"警告: 训练集特征维度 {len(X_train_processed)} 与标签维度 {len(y_train)} 不匹配，跳过该折")
                continue
                
            if len(X_val_processed) != len(y_val):
                print(f"警告: 验证集特征维度 {len(X_val_processed)} 与标签维度 {len(y_val)} 不匹配，跳过该折")
                continue
            
            try:
                # ========== 关键修复：修复神经网络参数 ==========
                # 在创建模型之前修复参数
                if params and model_type in ['fnn', 'deep_fnn', 'simple_fnn', 'resnet']:
                    # 创建参数的副本，避免修改原始参数
                    model_params = params.copy()
                    # 修复神经网络参数格式
                    model_params = self._fix_neural_network_params(model_params)
                else:
                    model_params = params.copy() if params else {}
                # ================================================
                
                # 创建和训练模型
                if model_type in ['fnn', 'deep_fnn', 'simple_fnn', 'resnet'] and self.nn_backend == 'pytorch':
                    # PyTorch模型
                    model = self.create_model_with_params(
                        model_type, model_params, X_train_processed.shape[1]
                    )
                    
                    # 计算batch_size，确保至少为2（如果样本数足够）
                    batch_size_ratio = model_params.get('batch_size_ratio', 0.1)
                    batch_size = max(1, int(len(X_train_processed) * batch_size_ratio))
                    # 保障batch_size≥2（类似PyTorch的处理）
                    if len(X_train_processed) >= 2:
                        batch_size = max(2, batch_size)
                    
                    # 训练
                    model.fit(
                        X_train_processed, y_train,
                        epochs=model_params.get('epochs', 1000),
                        batch_size=batch_size,
                        early_stopping=model_params.get('early_stopping', True),
                        patience=model_params.get('patience', 50),
                        verbose=False
                    )
                    
                    # 预测
                    y_val_pred = model.predict(X_val_processed)
                    
                elif model_type == 'gpr':
                    # 高斯过程回归特殊处理（可能内存消耗大）
                    try:
                        model = self.create_model_with_params(model_type, model_params)
                        model.fit(X_train_processed, y_train)
                        y_val_pred = model.predict(X_val_processed)
                    except MemoryError:
                        print("高斯过程回归内存不足，跳过此折")
                        continue
                elif model_type == 'poly':
                    # 多项式回归特殊处理（可能产生大量特征）
                    try:
                        model = self.create_model_with_params(model_type, model_params)
                        model.fit(X_train_processed, y_train)
                        y_val_pred = model.predict(X_val_processed)
                    except MemoryError:
                        print("多项式回归内存不足，跳过此折")
                        continue
                else:
                    # 其他模型
                    model = self.create_model_with_params(model_type, model_params)
                    X_train_model = prepare_model_input(model_type, X_train_processed)
                    X_val_model = prepare_model_input(model_type, X_val_processed)
                    model.fit(X_train_model, y_train)
                    y_val_pred = model.predict(X_val_model)
                
                # 计算指标
                mse_test = mean_squared_error(y_val, y_val_pred)
                mae_test = mean_absolute_error(y_val, y_val_pred)
                absolute_percent_errors_test = np.where(y_val != 0, np.abs((y_val - y_val_pred)/y_val), 0)
                mape_test = np.mean(absolute_percent_errors_test)

                r2_test = r2_score(y_val, y_val_pred)
                rmse_test = np.sqrt(mse_test)

                
                # 累加统计量
                total_rmse_test += rmse_test
                total_mae_test += mae_test
                total_r2_test += r2_test
                total_mape_test += mape_test
                total_folds += 1
                
                # 记录正R²的数量
                if r2_test >= 0:
                    positive_r2_count += 1
                
                # 更新最大R²
                if r2_test > max_r2:
                    max_r2 = r2_test
                    
            except Exception as e:
                print(f"模型训练或预测失败: {e}")
                continue
        
        # 计算平均统计量
        if total_folds > 0:
            average_rmse_test = total_rmse_test / total_folds
            average_r2_test = total_r2_test / total_folds
            average_mae_test = total_mae_test / total_folds
            average_mape_test = total_mape_test / total_folds
            pos_rate = positive_r2_count / total_folds
        else:
            average_r2_test = 0
            average_mae_test = 0
            average_rmse_test = 0 
            pos_rate = 0
            average_mape_test = 0 # 约束误差百分比
            max_r2 = 0  # 修复：total_folds==0时max_r2保持-inf会导致test_score=-inf，进而best_metrics永远为None

        percentage_error_rmse = average_rmse_test / abs(mean_property) if mean_property != 0 else 0 # 约束最大误差

        
        percentage_error_mae = average_mae_test / abs(mean_property) if mean_property != 0 else 0 # 约束误差总量百分比
        
        # 计算测试因子
        test_score = 6 * average_r2_test + 3 * pos_rate + 1 * max_r2 - 3 * percentage_error_mae - 2 * percentage_error_rmse - 5 * average_mape_test
        
        return test_score, average_r2_test, pos_rate, max_r2, percentage_error_mae, percentage_error_rmse, average_mape_test
    
    def calculate_test_with_hyperparam_search(self, X: pd.DataFrame, y: pd.Series, 
                                            model_type: str, n_iterations: int = 10, 
                                            n_folds: int = 5,
                                            seed_offset: int = 0) -> Tuple[float, float, float, float, float, float, float, Dict]:
        """进行超参数搜索，返回最佳测试因子和参数

        seed_offset: 随机搜索种子偏移（judge=1/2 每轮/每组合递增，避免不同特征子集
                     探索相同的超参组合与降维比例，降低局部最优风险）；贝叶斯路径忽略该参数。
        """
        
        print(f"\n使用 {self.search_method} 搜索方法进行超参数优化 (模型: {model_type})")
        
        if self.search_method == 'bayesian':
            if self.bayesian_searcher is None:
                print("警告: 贝叶斯搜索器未初始化，退回随机搜索")
                return self._random_hyperparam_search(X, y, model_type, n_iterations, n_folds, seed_offset)
            
            # 检查贝叶斯优化器是否已初始化该模型
            if (self.bayesian_searcher.bayesian_optimizer and 
                model_type in self.bayesian_searcher.bayesian_optimizer.optimizers):
                return self.bayesian_searcher.calculate_test_with_bayesian_search(
                    X, y, model_type, n_iterations, n_folds
                )
            else:
                print(f"模型 {model_type} 的贝叶斯优化器未初始化，退回随机搜索")
                return self._random_hyperparam_search(X, y, model_type, n_iterations, n_folds, seed_offset)
        else:
            print(f"执行随机搜索，迭代次数: {n_iterations}")
            return self._random_hyperparam_search(X, y, model_type, n_iterations, n_folds, seed_offset)

    def _calculate_additional_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """计算额外的性能指标"""
        import numpy as np
        
        metrics = {}
        
        # 1. 计算Pearson相关系数
        try:
            from scipy.stats import pearsonr
            pearson_corr, pearson_p_value = pearsonr(y_true.values, y_pred)
            metrics['pearson_corr'] = float(pearson_corr)
            metrics['pearson_p_value'] = float(pearson_p_value)
        except ImportError:
            # 如果没有scipy，使用numpy计算相关系数
            corr_matrix = np.corrcoef(y_true.values, y_pred)
            metrics['pearson_corr'] = float(corr_matrix[0, 1])
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
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(rmse)
        except Exception:
            metrics['rmse'] = None
        
        # 4. 计算MAE
        try:
            from sklearn.metrics import mean_absolute_error
            mae = mean_absolute_error(y_true, y_pred)
            metrics['mae'] = float(mae)
        except Exception:
            metrics['mae'] = None
        
        # 5. 计算R²
        try:
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            metrics['r2'] = float(r2)
        except Exception:
            metrics['r2'] = None
        
        return metrics

    def _get_back_feature_count(self, X: pd.DataFrame) -> int:
        """统计后部分特征（用于降维）的数量；无降维器时返回0"""
        if self.reducer is None:
            return 0
        try:
            _, back_features = self.reducer.split_features(X)
            return len(back_features)
        except Exception:
            return 0

    def _build_reduction_ratio_list(self, X: pd.DataFrame) -> List[float]:
        """按后部分特征量生成离散降维比例列表（0.05~0.95，含端点）。

        0<后部分特征<=50 → 步长0.1；50<后部分特征<=200 → 步长0.05；>200 → 步长0.03。
        none 模式（无降维器）由调用方处理为 [0.0]。
        """
        back_count = self._get_back_feature_count(X)
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
        return ratios

    def _random_hyperparam_search(self, X: pd.DataFrame, y: pd.Series, 
                                model_type: str, n_iterations: int = 10, 
                                n_folds: int = 5,
                                seed_offset: int = 0) -> Tuple[float, float, float, float, float, float, float, Dict]:
        """随机超参数搜索：降维比例独立于模型超参空间，避免笛卡尔积膨胀。

        降维比例按后部分特征量生成离散档位（0.05~0.95），与每个超参组合一一配对：
        不放回随机取，取空后随机种子自增进入下一轮，尽量保证组合与比例不重复。
        贝叶斯搜索仍沿用 param_space 中的原始比例空间（本方法只服务于随机搜索/贝叶斯兜底）。

        seed_offset: 随机种子偏移。judge=1/2 在特征子集变化时传入（按组合/轮次递增），
                     使不同特征子集探索不同的超参组合与降维比例配对，降低局部最优风险。
        """
        best_score = -float('inf')
        best_params = None
        best_metrics = None
        
        param_space = self.model_configs[model_type]['param_space']
        
        # ---- 降维比例独立处理 ----
        # 1) 增量 judge=0 等场景：param_space['reduction_ratio'] 被固定为单个值 → 全部组合使用该固定比例
        # 2) none 模式：无降维器 → 比例为0
        # 3) 其余：按后部分特征量生成离散比例列表
        ratio_config = param_space.get('reduction_ratio')
        if isinstance(ratio_config, list) and len(ratio_config) == 1:
            ratio_list = [float(ratio_config[0])]
            print(f"降维比例已固定为 {ratio_list[0]:.2f}（增量judge=0或外部固定）")
        elif self.reducer is None:
            ratio_list = [0.0]
        else:
            ratio_list = self._build_reduction_ratio_list(X)
            print(f"按后部分特征量({self._get_back_feature_count(X)}个)生成降维比例列表({len(ratio_list)}档): "
                  f"{', '.join(f'{r:.2f}' for r in ratio_list[:8])}{' ...' if len(ratio_list) > 8 else ''}")
        
        # 模型超参空间剔除降维比例，避免笛卡尔积膨胀（比例由上面的列表独立配对）
        from itertools import product
        model_param_space = {k: v for k, v in param_space.items() if k != 'reduction_ratio'}
        param_names = list(model_param_space.keys())
        param_values = list(model_param_space.values())
        all_possible_params = ([dict(zip(param_names, combo)) for combo in product(*param_values)]
                               if param_values else [{}])
        
        # 根据模型类型调整迭代次数
        model_list_low = [
            # 线性模型
            'linear', 'ridge', 'lasso', 'elasticnet', 'bayesian_ridge',
            # 简单树模型
            'dt',
            # 轻量SVM
            'linearsvr', 
            # 其他轻量模型
            'knn', 'huber', 'poly', 
        ]

        model_list_medium = [
            # SVM
            'svr', 'svr_rbf',
            # 基础树集成
            'rf', 'extra_trees', 'gbr', 'gbdt', 'hist_gbdt',
            # boost/核方法
            'adaboost', 'xgb', 'lgbm', 'catboost',
        ]

        model_list_high = [
            # 神经网络模型
            'fnn', 'deep_fnn', 
            'simple_fnn', 'resnet', 
            'gpr',
        ]
        
        if model_type in model_list_medium:
            n_iterations = max(1, round(n_iterations * 0.7))
        elif model_type in model_list_high:
            n_iterations = max(1, round(n_iterations * 0.5))
        
        max_possible_iterations = len(all_possible_params)
        
        if n_iterations > max_possible_iterations:
            n_iterations = max_possible_iterations
        
        if max_possible_iterations <= n_iterations:
            param_combinations = all_possible_params
        else:
            random.seed(42 + seed_offset)
            param_combinations = random.sample(all_possible_params, n_iterations)
        
        print(f"模型 {model_type}: 将测试 {len(param_combinations)}/{max_possible_iterations} 种参数组合")
        
        # 为每个超参组合分配降维比例：不放回随机取，取空后随机种子自增进入下一轮
        assigned_ratios = []
        ratio_seed = 42 + seed_offset
        ratio_pool = list(ratio_list)
        random.seed(ratio_seed)
        random.shuffle(ratio_pool)
        while len(assigned_ratios) < len(param_combinations):
            if not ratio_pool:
                ratio_seed += 10
                random.seed(ratio_seed)
                ratio_pool = list(ratio_list)
                random.shuffle(ratio_pool)
            assigned_ratios.append(ratio_pool.pop())

        for i, params in enumerate(param_combinations):
            # 合并降维比例到本次试验参数（最佳参数随比例一并保存，供贪心剔除/最终训练复用）
            trial_params = {**params, 'reduction_ratio': assigned_ratios[i]}
            print(f"  测试 {i+1}/{len(param_combinations)} - 参数: {trial_params}")
            
            # 修复：接收所有7个返回值
            score, avg_r2, pos_rate, max_r2, perc_error, percentage_error_rmse, average_mape_test = self.calculate_test_score(
                X, y, model_type, trial_params, n_folds
            )
            
            if score > best_score:
                best_score = score
                best_params = trial_params
                # 保存所有指标
                best_metrics = (avg_r2, pos_rate, max_r2, perc_error, percentage_error_rmse, average_mape_test)
            
            # 进度报告
            if (i + 1) % max(1, len(param_combinations) // 10) == 0:
                print(f"  进度: {i+1}/{len(param_combinations)} ({(i+1)/len(param_combinations)*100:.1f}%) \n 当前参数：{best_params}")
        
        # 返回所有指标
        if best_metrics is None:
            print(f"  警告: 模型 {model_type} 所有参数组合均失败，返回空结果跳过该模型")
            # 返回默认值让上层跳过该模型，不阻塞整个管道
            return -float('inf'), 0, 0, 0, 0, 0, 0, {}

        return best_score, best_metrics[0], best_metrics[1], best_metrics[2], best_metrics[3], best_metrics[4], best_metrics[5], best_params

    def _fix_neural_network_params(self, params):
        """修复神经网络参数格式"""
        if not params:
            return params
        
        # 检查是否是神经网络模型参数
        is_neural_net = any(key in params for key in ['hidden_layer_sizes', 
                                                    'hidden_layer_sizes_num_layers'])
        
        if is_neural_net and 'hidden_layer_sizes' not in params:
            # 从拆分参数重建
            if 'hidden_layer_sizes_num_layers' in params:
                num_layers = int(round(float(params.pop('hidden_layer_sizes_num_layers', 2))))
                first_layer = int(round(float(params.pop('hidden_layer_sizes_first_layer', 50))))
                # decay_rate 不存在时说明是等维架构（如ResNet），所有层统一大小
                decay_rate = float(params.pop('hidden_layer_sizes_decay_rate', 1.0)) if 'hidden_layer_sizes_decay_rate' in params else 1.0
                
                layers = []
                current_size = first_layer
                for _ in range(num_layers):
                    layer_size = int(round(current_size))
                    layer_size = max(5, layer_size)
                    layers.append(layer_size)
                    current_size = current_size * decay_rate
                
                params['hidden_layer_sizes'] = tuple(layers)
        
        return params
    
    def _get_all_possible_params(self, model_type: str) -> List[Dict]:
        """获取参数空间中所有可能的组合"""
        if hasattr(self, '_param_cache') and model_type in self._param_cache:
            return self._param_cache[model_type]
        
        from itertools import product
        
        param_space = self.model_configs[model_type]['param_space']
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        all_combinations = list(product(*param_values))
        all_params = [dict(zip(param_names, combo)) for combo in all_combinations]
        
        if not hasattr(self, '_param_cache'):
            self._param_cache = {}
        self._param_cache[model_type] = all_params
        
        print(f"模型 {model_type} 的参数空间大小: {len(all_params)}")
        
        return all_params
    
    def greedy_feature_selection_for_model(self, X: pd.DataFrame, y: pd.Series, 
                                        model_type: str, n_iterations: int = 10, 
                                        n_folds: int = 5) -> Dict:
        """对单个模型执行贪心特征选择"""
        X.columns = X.columns.astype(str)
        X = X.rename(columns=lambda x: str(x))
        
        if model_type not in self.model_configs:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        i = self.model_configs[model_type]['i']
        j = self.model_configs[model_type]['j']
        
        print(f"模型 {model_type} 的配置: i={i}, j={j}")
        
        # 分离前后部分特征
        if self.reducer:
            front_feature_list, back_feature_list = self.reducer.split_features(X)
        else:
            front_feature_list = list(X.columns)
            back_feature_list = []
        
        print(f"特征分配: 前部分特征={len(front_feature_list)}个, 后部分特征={len(back_feature_list)}个")
        
        # judge=2: 基于特征重要性的递归剔除（与贪心法并列的前部分特征筛选方案）
        if self.judge == 2 and len(front_feature_list) > 1:
            return self.importance_based_recursive_elimination(
                X, y, model_type, n_iterations, n_folds,
                front_feature_list=front_feature_list,
                back_feature_list=back_feature_list,
                max_no_improve_rounds=self.max_no_improve_rounds
            )

        # 初始测试 - 修复：接收所有返回值
        original_test, avg_r2, pos_rate, max_r2, perc_error, percentage_error_rmse, average_mape_test, best_params = self.calculate_test_with_hyperparam_search(
            X, y, model_type, n_iterations, n_folds
        )
        
        print(f'初始test因子: {original_test:.4f}, 平均R2: {avg_r2:.4f}, 正R2占比: {pos_rate:.4f}, 最大R2: {max_r2:.4f}, 误差量比: {perc_error:.4f}，均方误差量比：{percentage_error_rmse:.4f}，平均百分比误差: {average_mape_test:.4f}')
        
        # 贪心降维
        final_avg_r2_test = avg_r2
        final_pos_test = pos_rate
        final_max_test = max_r2
        final_per_Err_test = perc_error
        final_percentage_error_rmse = percentage_error_rmse  # 新增
        final_average_mape_test = average_mape_test  # 新增
        final_best_params = best_params
        
        if self.judge == 1 and len(front_feature_list) > 0:
            initial = 1
            while True:
                print(f"当前前部分特征数量: {len(front_feature_list)}")
                print(f"配置参数: 每次剔除 {i} 个前部分特征，重复 {j} 次")
                
                feature_combinations = self.generate_unique_feature_combinations(
                    front_feature_list, i, j, seed_offset=initial*10, model_type=model_type
                )
                
                if not feature_combinations:
                    print("前部分特征数量不足，无法继续剔除，停止降维")
                    break
                
                print(f"生成了 {len(feature_combinations)} 个不同的前部分特征组合进行测试")
                
                best_test = -float('inf')
                best_features_to_remove = None
                best_metrics = None
                best_params_this_round = None
                
                for combo_idx, features_to_remove in enumerate(feature_combinations):
                    X_temp = X.drop(columns=features_to_remove)
                    
                    # 修复：接收所有返回值
                    # 种子与特征组合挂钩：同一轮内各组合、不同轮之间探索不同的超参/降维比例，避免局部最优
                    test, avg_r2, pos, max_r, perc_e, perc_rmse, avg_mape, params = self.calculate_test_with_hyperparam_search(
                        X_temp, y, model_type, n_iterations, n_folds,
                        seed_offset=initial * 10 + combo_idx
                    )
                    
                    print(f"  尝试剔除前部分特征组合 {combo_idx+1}/{len(feature_combinations)}: {features_to_remove}")
                    print(f"  当前指标 - R2: {avg_r2:.4f}, 误差比: {perc_e:.4f}, RMSE比: {perc_rmse:.4f}, MAPE: {avg_mape:.4f}")
                    
                    if test > best_test:
                        best_test = test
                        best_avg_r2 = avg_r2
                        best_pos = pos
                        best_max = max_r
                        best_per_e = perc_e
                        best_perc_rmse = perc_rmse  # 新增
                        best_avg_mape = avg_mape  # 新增
                        best_features_to_remove = features_to_remove
                        best_params_this_round = params
                
                if best_test <= original_test:
                    print(f"第{initial}轮结束，无法提升test因子，停止降维")
                    break
                else:
                    # 更新特征列表
                    for feature in best_features_to_remove:
                        front_feature_list.remove(feature)
                        X = X.drop(columns=[feature])
                    
                    # 更新所有变量
                    original_test = best_test
                    final_avg_r2_test = best_avg_r2
                    final_pos_test = best_pos
                    final_max_test = best_max
                    final_per_Err_test = best_per_e
                    final_percentage_error_rmse = best_perc_rmse  # 新增
                    final_average_mape_test = best_avg_mape  # 新增
                    final_best_params = best_params_this_round
                    
                    print(f'第{initial}轮剔除的前部分特征: {best_features_to_remove}')
                    print(f'正R2占比：{best_pos:.4f}，最大R2：{best_max:.4f}，平均R2：{best_avg_r2:.4f}')
                    print(f'误差量比：{best_per_e:.4f}，RMSE误差比：{best_perc_rmse:.4f}，MAPE：{best_avg_mape:.4f}，test因子：{best_test:.4f}')
                    initial += 1
        
        # 构建最终结果
        if self.reducer:
            final_features = front_feature_list + back_feature_list
            display_features = front_feature_list + [f'{self.reducer.reduction_type}_主成分{i+1}' 
                                                for i in range(len(back_feature_list))]
        else:
            final_features = front_feature_list
            display_features = front_feature_list
        
        return {
            'model_type': model_type,
            'final_features': display_features,
            'front_features': front_feature_list,
            'back_features': back_feature_list,
            'num_features': len(display_features),
            'test_score': original_test,
            'avg_r2': final_avg_r2_test,
            'pos_rate': final_pos_test,
            'max_r2': final_max_test,
            'perc_error': final_per_Err_test,
            'percentage_error_rmse': final_percentage_error_rmse,  # 新增
            'average_mape_test': final_average_mape_test,  # 新增
            'best_params': final_best_params,
            'X_final': X[final_features] if len(final_features) > 0 else pd.DataFrame()
        }

    def importance_based_recursive_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                               model_type: str, n_iterations: int = 10, 
                                               n_folds: int = 5,
                                               front_feature_list: Optional[List[str]] = None,
                                               back_feature_list: Optional[List[str]] = None,
                                               max_no_improve_rounds: int = 0) -> Dict:
        """基于特征重要性的递归剔除（judge=2）

        每轮流程：超参搜索取最优(test因子) → 用最优参数完整训练 → 计算特征重要性 →
        剔除前部分特征中重要性最低的1个。若剔除后 test因子未提升则进入容忍期：
        允许继续剔除 max_no_improve_rounds 个（无提升也继续），超过后停止并回退到
        历史最佳特征集；该轮数不能超过剩余特征总数。
        只对前部分特征做剔除，与贪心法一致，但按重要性逐个剔除、可解释性更强。
        """
        X.columns = X.columns.astype(str)
        X = X.rename(columns=lambda x: str(x))

        if front_feature_list is None or back_feature_list is None:
            if self.reducer:
                front_feature_list, back_feature_list = self.reducer.split_features(X)
            else:
                front_feature_list = list(X.columns)
                back_feature_list = []
        front_feature_list = list(front_feature_list)
        back_feature_list = list(back_feature_list)

        # 容忍轮数不能超过剩余特征总数（每轮剔除1个，至少保留1个前部分特征）
        max_no_improve_rounds = max(0, int(max_no_improve_rounds or 0))
        max_no_improve_rounds = min(max_no_improve_rounds, max(0, len(front_feature_list) - 1))

        print(f"\n模型 {model_type} 使用 judge=2 基于特征重要性的递归剔除")
        print(f"特征分配: 前部分特征={len(front_feature_list)}个, 后部分特征={len(back_feature_list)}个, "
              f"无提升容忍轮数={max_no_improve_rounds}")

        # 初始超参搜索（完整特征集）
        original_test, avg_r2, pos_rate, max_r2, perc_error, percentage_error_rmse, average_mape_test, best_params = \
            self.calculate_test_with_hyperparam_search(X, y, model_type, n_iterations, n_folds)
        print(f'初始test因子: {original_test:.4f}, 平均R2: {avg_r2:.4f}, 正R2占比: {pos_rate:.4f}, '
              f'最大R2: {max_r2:.4f}, 误差量比: {perc_error:.4f}，均方误差量比：{percentage_error_rmse:.4f}，平均百分比误差: {average_mape_test:.4f}')

        final_avg_r2_test = avg_r2
        final_pos_test = pos_rate
        final_max_test = max_r2
        final_per_Err_test = perc_error
        final_percentage_error_rmse = percentage_error_rmse
        final_average_mape_test = average_mape_test
        final_best_params = best_params

        # 历史最佳状态（用于容忍期结束后回退）
        best_test = original_test
        best_X = X.copy()
        best_front = list(front_feature_list)

        elimination_history = []
        round_no = 0
        no_improve_count = 0

        while len(front_feature_list) > 1:
            round_no += 1
            print(f"\n[重要度剔除 第{round_no}轮] 当前前部分特征数: {len(front_feature_list)}")

            # 1) 用当前最优参数完整训练（与真实管线一致：先降维再训练），计算特征重要性
            try:
                model, X_processed = self._fit_full_model_for_importance(X, y, model_type, final_best_params)
                # 只计算剩余前部分特征的重要性：主成分列不参与剔除，跳过其置换开销；
                # 模型仍在完整输入空间(前部分+主成分)上计算，前部分特征的数值不受影响
                front_cols = [c for c in front_feature_list if c in X_processed.columns]
                importance = compute_feature_importance(model, X_processed, y, model_type, features=front_cols)
                if importance is None or importance.sum() == 0:
                    print("  无法计算有效特征重要性，停止剔除")
                    break
            except Exception as e:
                print(f"  重要性计算失败: {e}，停止剔除")
                break

            # 2) 仅在前部分特征范围内选择重要性最低的特征
            front_imp = importance.reindex([c for c in front_feature_list if c in importance.index])
            if front_imp.empty:
                print("  前部分特征为空，停止剔除")
                break
            worst_feature = front_imp.idxmin()
            worst_imp = float(front_imp[worst_feature])
            print(f"  剔除候选: {worst_feature} (归一化重要性={worst_imp:.6f})")

            # 3) 剔除后重新超参搜索，评估test因子
            X_candidate = X.drop(columns=[worst_feature])
            # 种子与轮次挂钩：每轮剔除后重新探索不同的超参/降维比例，避免局部最优
            test, avg_r2, pos, max_r, perc_e, perc_rmse, avg_mape, params = \
                self.calculate_test_with_hyperparam_search(X_candidate, y, model_type, n_iterations, n_folds,
                                                           seed_offset=round_no)

            improved = test > original_test
            if improved:
                # 4a) 提升：接受剔除并更新历史最佳状态
                no_improve_count = 0
                X = X_candidate
                front_feature_list.remove(worst_feature)
                original_test = test
                final_avg_r2_test = avg_r2
                final_pos_test = pos
                final_max_test = max_r
                final_per_Err_test = perc_e
                final_percentage_error_rmse = perc_rmse
                final_average_mape_test = avg_mape
                final_best_params = params
                best_test = test
                best_X = X.copy()
                best_front = list(front_feature_list)
                elimination_history.append({
                    'round': round_no,
                    'removed_feature': worst_feature,
                    'importance': worst_imp,
                    'test_score': test,
                    'improved': True,
                    'avg_r2': avg_r2,
                    'avg_mape': avg_mape,
                    'remaining_front_features': len(front_feature_list)
                })
                print(f"  第{round_no}轮剔除前部分特征: {worst_feature}，test因子: {test:.4f}（提升），平均R2: {avg_r2:.4f}")
            else:
                # 4b) 未提升：进入容忍期（未达上限则继续尝试，不更新original_test）
                no_improve_count += 1
                print(f"  第{round_no}轮剔除[{worst_feature}]后 test因子 {test:.4f} <= {original_test:.4f}，未提升 "
                      f"(无提升计数 {no_improve_count}/{max_no_improve_rounds})")
                if no_improve_count > max_no_improve_rounds:
                    print(f"  达到无提升容忍上限({max_no_improve_rounds})，停止剔除，回退到最佳特征集")
                    break
                X = X_candidate
                front_feature_list.remove(worst_feature)
                elimination_history.append({
                    'round': round_no,
                    'removed_feature': worst_feature,
                    'importance': worst_imp,
                    'test_score': test,
                    'improved': False,
                    'avg_r2': avg_r2,
                    'avg_mape': avg_mape,
                    'remaining_front_features': len(front_feature_list)
                })

        # 回退到历史最佳特征集（容忍期内可能剔掉了有效特征）
        if best_front != front_feature_list:
            X = best_X
            front_feature_list = best_front
            original_test = best_test
            print(f"回退到最佳特征集: test因子={best_test:.4f}, 前部分特征={len(best_front)}个")

        if elimination_history:
            print(f"\n剔除历史（共{len(elimination_history)}轮）:")
            for h in elimination_history:
                tag = "提升" if h['improved'] else "未提升"
                print(f"  第{h['round']}轮: 剔除 {h['removed_feature']} (重要性={h['importance']:.6f}), "
                      f"test因子={h['test_score']:.4f} [{tag}], 剩余前部分特征={h['remaining_front_features']}")

        # 构建最终结果（与贪心法结构一致）
        if self.reducer:
            final_features = front_feature_list + back_feature_list
            display_features = front_feature_list + [f'{self.reducer.reduction_type}_主成分{i+1}' 
                                                     for i in range(len(back_feature_list))]
        else:
            final_features = front_feature_list
            display_features = front_feature_list

        return {
            'model_type': model_type,
            'final_features': display_features,
            'front_features': front_feature_list,
            'back_features': back_feature_list,
            'num_features': len(display_features),
            'test_score': original_test,
            'avg_r2': final_avg_r2_test,
            'pos_rate': final_pos_test,
            'max_r2': final_max_test,
            'perc_error': final_per_Err_test,
            'percentage_error_rmse': final_percentage_error_rmse,
            'average_mape_test': final_average_mape_test,
            'best_params': final_best_params,
            'elimination_history': elimination_history,
            'X_final': X[final_features] if len(final_features) > 0 else pd.DataFrame()
        }

    def _fit_full_model_for_importance(self, X: pd.DataFrame, y: pd.Series, 
                                       model_type: str, params: Optional[Dict] = None):
        """用当前最优参数在完整数据集上训练模型（仅用于计算特征重要性，不保存）

        与真实管线保持一致：先对后部分特征按 reduction_ratio 降维（前部分原样保留），
        再在 前部分+主成分 的特征空间上训练，确保重要性在模型真实输入空间上计算。
        返回 (model, X_processed)，X_processed 为降维后的特征矩阵（importance 需在其上计算）。
        """
        params = dict(params) if params else {}
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        reduction_ratio = params.get('reduction_ratio', 0.5)
        if self.reducer:
            X_processed, _, _ = self.reducer.reduce_fold(X, X, y, ratio=reduction_ratio)
        else:
            X_processed = X
        input_size = X_processed.shape[1]

        model = self.create_model_with_params(model_type, params, input_size)
        if model_type in ['fnn', 'deep_fnn', 'simple_fnn', 'resnet'] and self.nn_backend == 'pytorch':
            batch_size = max(1, int(len(X_processed) * params.get('batch_size_ratio', 0.1)))
            model.fit(X_processed, y,
                      epochs=params.get('epochs', 500),
                      batch_size=batch_size,
                      early_stopping=False,
                      patience=params.get('patience', 50),
                      verbose=False)
        else:
            model.fit(prepare_model_input(model_type, X_processed), y)
        return model, X_processed

    def generate_unique_feature_combinations(self, feature_list: List[str], i: int, j: int, 
                                           seed_offset: int = 0, model_type: Optional[str] = None) -> List[List[str]]:
        """生成唯一的特征组合用于剔除"""
        n_features = len(feature_list)
        
        if n_features <= i:
            return []
        
        base_seed = 42
        if model_type:
            model_hash = hash(model_type) % 1000
            current_seed = base_seed + seed_offset + model_hash
        else:
            current_seed = base_seed + seed_offset
        
        random.seed(current_seed)
        total_combinations = len(list(combinations(feature_list, i)))
        
        if total_combinations <= j:
            combinations_list = list(combinations(feature_list, i))
            random.shuffle(combinations_list)
            return [list(combo) for combo in combinations_list]
        else:
            selected_combinations = set()
            while len(selected_combinations) < j:
                combo = tuple(sorted(random.sample(feature_list, i)))
                selected_combinations.add(combo)
            
            return [list(combo) for combo in selected_combinations]
        
    def train_final_model(self, output_dir, training_params_dir, X: pd.DataFrame, y: pd.Series, model_type: str,
                        front_features: List[str], back_features: List[str],
                        best_params: Dict, property_name: str = "目标性质",
                        additional_save_path: Optional[str] = None) -> Optional[Dict]:
        """训练最终模型"""
        # from sklearn.preprocessing import StandardScaler
        
        # 合并特征
        feature_list = front_features + back_features
        X_selected = X[feature_list]
        
        print(f"\n训练最终模型 - 性质: {property_name}")
        print(f"特征数量: {len(feature_list)} (前部分: {len(front_features)}, 后部分: {len(back_features)})")
        
        # 对完整训练集进行标准化
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X_selected)
        # X_scaled = pd.DataFrame(
        #     X_scaled, 
        #     columns=X_selected.columns, 
        #     index=X_selected.index
        # )
        X_scaled = X_selected
        
        # print(f"数据标准化完成 (均值: {scaler.mean_[0]:.2f}, 方差: {scaler.var_[0]:.2f})")
        
        # 保存标准化器（可选，如果后续需要用于测试集）
        safe_prop = property_name.replace("（", "_").replace("）", "_").replace("/", "_")
        # scaler_filename = f"{self.best_model_dir}/scaler_{safe_prop}.pkl"
        # joblib.dump(scaler, scaler_filename)
        # print(f"标准化器已保存至：{scaler_filename}")
        
        # 应用降维（如果有）
        if self.reducer:
            reduction_ratio = best_params.get('reduction_ratio', 0.5) if best_params else 0.5
            
            print(f"应用{self.reducer.reduction_type.upper()}降维，降维比例: {reduction_ratio}")
            
            # 使用标准化后的数据进行降维
            # 注意：我们需要修改降维器，使其可以禁用内部标准化
            # 假设降维器已添加skip_standardization参数
            X_final, reduction_info = self.reducer.reduce_full_data(
                output_dir, training_params_dir, X_scaled, y, ratio=reduction_ratio, property_name=property_name
            )
            if reduction_info is None:
                print("后部分为空，本次模型按无降维模式保存")
        else:
            X_final = X_scaled
            reduction_info = None
            print("不使用降维")
        
        # 确保索引对齐
        X_final = X_final.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        try:
            # 创建模型
            if model_type in ['fnn', 'deep_fnn', 'simple_fnn', 'resnet'] and self.nn_backend == 'pytorch':
                # PyTorch模型
                input_size = X_final.shape[1]
                model = self.create_model_with_params(model_type, best_params, input_size)
                
                print(f"使用完整训练集数据训练最终PyTorch模型...")
                print(f"训练数据形状: X_final={X_final.shape}, y={y.shape}")
                
                # 计算batch_size
                batch_size_ratio = best_params.get('batch_size_ratio', 0.1)
                batch_size = max(1, int(len(X_final) * batch_size_ratio))
                
                # 训练
                model.fit(
                    X_final, y,
                    epochs=best_params.get('epochs', 1000),
                    batch_size=batch_size,
                    early_stopping=best_params.get('early_stopping', True),
                    patience=best_params.get('patience', 50),
                    verbose=True
                )
                
                # 预测
                y_train_pred = model.predict(X_final)
                
                # 保存PyTorch模型
                model_save_path = f"{self.best_model_dir}/best_model_{safe_prop}.pth"
                self._print_device_info_()
                model_config = {
                    'model_type': model_type,
                    'input_size': input_size,
                    'hidden_sizes': best_params.get('hidden_layer_sizes', (50,)),
                    'activation': best_params.get('activation', 'relu'),
                    'dropout_rate': best_params.get('dropout_rate', 0.0),
                    'batch_norm': best_params.get('batch_norm', False)
                }
                
                PyTorchModelSaver.save_model(model.model, model_save_path, model_config)
                print(f"PyTorch模型已保存到 {model_save_path}")
                
            else:
                # 其他模型（sklearn等）
                final_params = best_params.copy() if best_params else {}
                final_params.pop('reduction_ratio', None)
                final_params = self._fix_neural_network_params(final_params)

                if model_type in ['fnn', 'deep_fnn', 'simple_fnn', 'resnet'] and 'batch_size_ratio' in final_params:
                    batch_size_ratio = final_params.pop('batch_size_ratio')
                    n_samples = len(X_final)
                    
                    if final_params.get('early_stopping', False):
                        val_frac = final_params.get('validation_fraction', 0.1)
                        effective_samples = int(n_samples * (1 - val_frac))
                    else:
                        effective_samples = n_samples
                    
                    batch_size = max(1, int(effective_samples * batch_size_ratio))
                    batch_size = min(batch_size, effective_samples)
                    # 保障batch_size≥2（类似PyTorch的处理）
                    if effective_samples >= 2:
                        batch_size = max(2, batch_size)
                    final_params['batch_size'] = batch_size
                
                # KNN安全截断：n_neighbors不能超过训练样本数
                if model_type == 'knn' and 'n_neighbors' in final_params:
                    n_samples_final = len(y)
                    if final_params['n_neighbors'] > n_samples_final:
                        print(f"  [KNN安全] n_neighbors从{final_params['n_neighbors']}截断为{n_samples_final}（样本数限制）")
                        final_params['n_neighbors'] = n_samples_final

                model = self.create_model_with_params(model_type, final_params)
                print(f"使用完整训练集数据训练最终模型...")
                print(f"训练数据形状: X_final={X_final.shape}, y={y.shape}")
                
                X_final_model = prepare_model_input(model_type, X_final)
                model.fit(X_final_model, y)
                y_train_pred = model.predict(X_final_model)
                
                # 保存sklearn模型
                model_filename = f"{self.best_model_dir}/best_model_{safe_prop}.pkl"
                joblib.dump(model, model_filename)
                print(f"最佳模型已导出至：{model_filename}")
            
            # 计算性能指标
            train_r2 = r2_score(y, y_train_pred)
            train_mse = mean_squared_error(y, y_train_pred)
            train_mae = mean_absolute_error(y, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            
            mean_property = y.mean()
            percentage_error = train_mae / abs(mean_property) if mean_property != 0 else 0
            
            # print(f"\n训练集性能指标:")
            # print(f"R²: {train_r2:.4f}")
            # print(f"MSE: {train_mse:.4f}")
            # print(f"RMSE: {train_rmse:.4f}")
            # print(f"MAE: {train_mae:.4f}")
            # print(f"平均相对误差: {percentage_error:.4f}")
            
            # 计算额外指标
            additional_metrics = self._calculate_additional_metrics(y, y_train_pred)
            if additional_metrics:
                print("训练指标:")
                for metric_name, metric_value in additional_metrics.items():
                    if metric_value is not None:
                        print(f"  {metric_name}: {metric_value:.4f}")
            
            # 保存结果
            try:
                self._save_training_results(output_dir, property_name, y, y_train_pred, train_r2, 
                                        train_rmse, train_mae, percentage_error, additional_metrics)
            except Exception as save_error:
                print(f"保存训练结果时出错: {save_error}")
            
            # 保存模型信息
            reduction_applied = self.reducer is not None and reduction_info is not None
            reduction_type = self.reducer.reduction_type if reduction_applied else 'none'
            reduction_ratio = best_params.get('reduction_ratio') if reduction_applied else None
            
            model_info = {
                'property': property_name,
                'best_model': model_type,
                'num_features': X_final.shape[1],
                'features': X_final.columns.tolist(),
                'hyperparameters': best_params,
                'train_r2': train_r2,
                # 'train_mae': train_mae,
                # 'train_rmse': train_rmse,
                # 'train_mse': train_mse,
                'mean_property': float(mean_property),
                'percentage_error': float(percentage_error),
                'reduction_type': reduction_type,
                'reduction_ratio': reduction_ratio,
                'reduction_features': self.reduction_features if reduction_applied else None,
                'orig_features': self.orig_features,
                'nn_backend': self.nn_backend,
            }
            
            return {
                'model': model,
                'train_r2': train_r2,
                'train_mse': train_mse,
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'percentage_error': percentage_error,
                'y_train_pred': y_train_pred,
                'y_train_pred_true': y,  # 保存实际值
                'reduction_info': reduction_info,
                'final_features': X_final.columns.tolist(),
                'model_info': model_info,
                'train_eval_metrics': additional_metrics
            }
            
        except Exception as e:
            print(f"最终模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_training_results(self, output_dir, property_name: str, y_true: pd.Series, 
                            y_pred: np.ndarray, r2: float, rmse: float, 
                            mae: float, perc_error: float, additional_metrics: Dict = None):
        """保存训练结果（构建内存字典，最终由optimize_all_models写入JSON）"""
        def _make_serializable(val):
            if isinstance(val, (np.integer,)):
                return int(val)
            if isinstance(val, (np.floating,)):
                if np.isnan(val) or np.isinf(val):
                    return None
                return float(val)
            if isinstance(val, np.ndarray):
                return [_make_serializable(v) for v in val.tolist()]
            if isinstance(val, (list, tuple)):
                return [_make_serializable(v) for v in val]
            return val

        safe_prop = property_name.replace("（", "_").replace("）", "_").replace("/", "_")
        
        predictions = []
        for exp_val, pred_val in zip(y_true.values, y_pred):
            predictions.append({
                f'{property_name}_exp': _make_serializable(exp_val),
                f'{property_name}_pred': _make_serializable(pred_val)
            })
        
        metrics = {
            'R²': _make_serializable(r2),
            'RMSE': _make_serializable(rmse),
            'MAE': _make_serializable(mae),
            '平均相对误差(%)': _make_serializable(perc_error * 100)
        }
        
        if additional_metrics:
            for metric_name, metric_value in additional_metrics.items():
                if metric_value is not None:
                    if metric_name == 'pearson_corr':
                        metrics['Pearson相关系数'] = _make_serializable(metric_value)
                    elif metric_name == 'mape':
                        metrics['MAPE(%)'] = _make_serializable(metric_value)
                    elif metric_name == 'pearson_p_value':
                        metrics['Pearson p值'] = _make_serializable(metric_value)
        
        self._best_model_perform_data[safe_prop] = {
            'property_name': property_name,
            'predictions': predictions,
            'metrics': metrics
        }
        
        print(f"训练结果已记录到内存 (性质: {property_name})")

    def _save_perform_json(self, output_dir: str):
        """将内存中的模型性能数据保存为JSON文件"""
        if not self._best_model_perform_data:
            return
        perform_json_file = f"{output_dir}/best_model_perform.json"
        try:
            existing_data = {}
            if os.path.exists(perform_json_file):
                with open(perform_json_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            existing_data.update(self._best_model_perform_data)
            with open(perform_json_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            print(f"模型性能数据已保存到 '{perform_json_file}'")
        except Exception as e:
            print(f"保存模型性能JSON失败: {e}")

    def optimize_all_models(self, output_dir, training_params_dir: str, X: pd.DataFrame, y: pd.Series, prop: str, 
                        model_types: Optional[List[str]] = None, n_iterations: int = 10, 
                        n_folds: int = 5, additional_save_path: Optional[str] = None) -> Tuple[Dict, Dict, Optional[Dict]]:
        """优化所有模型"""
        print(f"开始优化所有模型 - 目标: {prop}, 前部分特征={self.orig_features}, 降维器={self.reducer.reduction_type if self.reducer else 'none'}")
        
        if model_types is None:
            model_types = list(self.model_configs.keys())
        
        results = {}
        best_model_info = None
        best_test_score = -float('inf')
        best_model_type = None
        
        for model_type in model_types:
            print(f"\n{'='*80}")
            print(f"开始处理模型: {model_type}（评估属性：{prop}）")
            print(f"{'='*80}")
            
            if not self.model_configs[model_type]['param_space']:
                print(f"跳过{model_type}: 参数空间为空")
                continue
            
            feature_selection_result = self.greedy_feature_selection_for_model(
                X.copy(), y, model_type, n_iterations, n_folds
            )
            
            results[model_type] = feature_selection_result
            
            current_score = feature_selection_result['test_score']
            print(f"模型 {model_type} 的test_score: {current_score:.4f}")
            
            if current_score > best_test_score:
                best_test_score = current_score
                best_model_type = model_type
                best_model_info = {
                    'model_type': model_type,
                    'feature_selection_result': feature_selection_result
                }
        
        # 训练最佳模型
        if best_model_info:
            model_type = best_model_info['model_type']
            feature_selection_result = best_model_info['feature_selection_result']
            
            print(f"\n{'*'*80}")
            print(f"对最佳模型 {model_type} 进行完整训练 (test因子: {best_test_score:.4f})")
            print(f"{'='*80}")
            
            final_model_result = self.train_final_model(
                output_dir,
                training_params_dir,
                X=X, 
                y=y,
                model_type=model_type, 
                front_features=feature_selection_result['front_features'],
                back_features=feature_selection_result['back_features'],
                best_params=feature_selection_result['best_params'],
                property_name=prop,
                additional_save_path=additional_save_path
            )
            
            if final_model_result:
                # 注入交叉验证指标：最佳参数组合的交叉验证平均R2与平均MAPE
                final_model_result['model_info']['cv_avg_r2'] = feature_selection_result.get('avg_r2')
                final_model_result['model_info']['cv_avg_mape'] = feature_selection_result.get('average_mape_test')
                # 保存模型（传入额外保存路径）
                self._save_best_model(final_model_result, prop, best_test_score, 
                                    feature_selection_result, additional_save_path)
                results[model_type] = {**feature_selection_result, **final_model_result}
                
                # 更新模型信息中的test_score
                final_model_result['model_info']['test_score'] = best_test_score
                self._save_perform_json(output_dir)
                return results, {prop: {model_type: results[model_type]}}, final_model_result['model_info']
        
        self._save_perform_json(output_dir)
        return results, {prop: results}, None

    def _save_best_model(self, final_model_result: Dict, prop: str, 
                        best_test_score: float, feature_selection_result: Dict,
                        additional_save_path: Optional[str] = None):
        """保存最佳模型
        
        Args:
            final_model_result: 最终模型结果
            prop: 性质名称
            best_test_score: 最佳测试分数
            feature_selection_result: 特征选择结果
            additional_save_path: 额外的保存路径（可选）
        """
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)
        
        safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")
        
        # 保存模型.pkl文件（用于预测）
        if not (final_model_result['model_info']['nn_backend'] == 'pytorch'):
            model_filename = f"{self.best_model_dir}/best_model_{safe_prop}.pkl"
            joblib.dump(final_model_result['model'], model_filename)
            print(f" 最佳模型已导出至：{model_filename}")
        
        # 保存特征列表.json
        feature_filename = f"{self.best_model_dir}/best_features_{safe_prop}.json"
        with open(feature_filename, 'w', encoding='utf-8') as f:
            json.dump(final_model_result['final_features'], f, ensure_ascii=False, indent=2)
        print(f" 最佳模型特征列表已保存至：{feature_filename}")
        
        # 保存超参数.json（仅用于记录）
        params_filename = f"{self.best_model_dir}/best_params_{safe_prop}.json"
        
        # 简单清理参数以便JSON保存
        cleaned_params = self._clean_params_for_json(feature_selection_result['best_params'])
        
        with open(params_filename, 'w', encoding='utf-8') as f:
            json.dump(cleaned_params, f, ensure_ascii=False, indent=2)
        print(f" 最佳模型超参数已保存至：{params_filename}")
        
        # 计算额外性能指标
        y_true = final_model_result.get('y_train_pred_true', None)
        y_pred = final_model_result.get('y_train_pred', None)
        additional_metrics = {}
        
        if y_true is not None and y_pred is not None:
            additional_metrics = self._calculate_additional_metrics(y_true, y_pred)
        
        # 保存模型信息.json（包含额外指标）
        model_info = final_model_result['model_info']
        model_info.update({
            'test_score': best_test_score,
            'num_features': len(final_model_result['final_features']),
            'features': final_model_result['final_features'],
            'train_eval_metrics': additional_metrics,  # 添加额外指标
        })
        
        # 添加训练集的预测值和实际值（用于后续分析）
        # if 'y_train_pred' in final_model_result and 'y_train_pred_true' in final_model_result:
        #     model_info['y_train_true'] = final_model_result['y_train_pred_true'].tolist()
        #     model_info['y_train_pred'] = final_model_result['y_train_pred'].tolist()
        
        # 清理模型信息中的参数
        if 'hyperparameters' in model_info:
            model_info['hyperparameters'] = self._clean_params_for_json(model_info['hyperparameters'])
        
        # 保存到默认路径
        info_filename = f"{self.best_model_dir}/best_model_info_{safe_prop}.json"
        with open(info_filename, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print(f" 最佳模型信息已保存至：{info_filename}")
        
        # 如果有额外的保存路径，也保存一份
        if additional_save_path and os.path.exists(additional_save_path):
            try:
                additional_save_path = additional_save_path
                os.makedirs(additional_save_path, exist_ok=True)
                
                # 保存JSON文件
                additional_info_filename = f"{additional_save_path}/best_model_info_{safe_prop}.json"
                with open(additional_info_filename, 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, ensure_ascii=False, indent=2)
                print(f" 最佳模型信息已保存至额外路径：{additional_info_filename}")
                
                # 保存模型文件（可选）
                # model_src = f"{self.best_model_dir}/best_model_{safe_prop}.pkl"
                # model_dst = f"{additional_save_path}/best_model_{safe_prop}.pkl"
                # if os.path.exists(model_src):
                #     shutil.copy2(model_src, model_dst)
                #     print(f" 模型文件已复制到额外路径：{model_dst}")
                    
            except Exception as e:
                print(f" 保存到额外路径失败：{e}")
        else:
            if not additional_save_path:
                print(f" 额外保存路径不存在，跳过保存到该路径：{additional_save_path}")

    def _clean_params_for_json(self, params: Dict) -> Dict:
        """清理参数以便JSON保存，仅用于记录目的"""
        import numpy as np
        import json
        
        if params is None:
            return {}
        
        def convert_value(v):
            # 基本类型直接返回
            if v is None or isinstance(v, (bool, int, float, str)):
                return v
            
            # 处理numpy类型
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
            
            # 处理列表/元组
            if isinstance(v, (list, tuple)):
                return [convert_value(item) for item in v]
            
            # 处理字典
            if isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            
            # 其他类型：转换为字符串
            try:
                return str(v)
            except:
                return f"<无法序列化的对象: {type(v)}>"
        
        return convert_value(params)

    def run_plus_mode(self, output_dir, training_params_dir, X: pd.DataFrame, y: pd.Series, prop: str,
                    model_list: List[str], n_iterations: int, n_folds: int,
                    custom_modes: Optional[List[str]] = None,
                    additional_save_path: Optional[str] = None) -> Dict:
        """运行 PLUS/CUSTOM 模式并比较多种降维方式。"""
        
        # 确定要比较的模式列表
        if custom_modes:
            modes = custom_modes if custom_modes else ['pca', 'pls', 'svd']  # 使用传入的自定义模式列表
            print(f"\n{'='*80}")
            print(f"开始CUSTOM模式: 比较 {len(modes)} 种降维方式")
            print(f"模式列表: {', '.join(modes)}")
            print(f"{'='*80}")
        else:
            modes = ['pca', 'pls', 'svd']  # 默认三模式
            print(f"\n{'='*80}")
            print(f"开始PLUS模式: 比较pca、pls、svd三种降维方式")
            print(f"{'='*80}")
        
        # 检查可用模块
        if PCAReducer is None:
            if 'pca' in modes:
                modes.remove('pca')
                print("警告: PCA模块不可用，跳过PCA模式")
        
        if PLSReducer is None:
            if 'pls' in modes:
                modes.remove('pls')
                print("警告: PLS模块不可用，跳过PLS模式")
        
        if SVDReducer is None:
            if 'svd' in modes:
                modes.remove('svd')
                print("警告: SVD模块不可用，跳过SVD模式")
            if 'tsvd' in modes:
                modes.remove('tsvd')
                print("警告: SVD模块不可用，跳过TSVD模式")
        
        if len(modes) == 0:
            print("错误: 没有可用的降维模式，模式比较无法运行")
            return {}
        
        # 初始化模式结果字典
        mode_results = {}
        
        # 分别运行每种模式
        for mode in modes:
            print(f"\n{'='*60}")
            print(f"运行 {mode.upper()} 模式")
            print(f"{'='*60}")
            
            # 创建临时目录
            temp_dir = f"{output_dir}/temp_{mode}_{prop}"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            try:
                # 创建优化器实例
                optimizer = ModelOptimizer()
                optimizer.property_names = [prop]
                optimizer.best_model_dir = temp_dir
                optimizer.judge = self.judge
                optimizer.max_no_improve_rounds = self.max_no_improve_rounds
                optimizer.nn_backend = self.nn_backend
                optimizer.nn_config = self.nn_config
                optimizer.reduction_features = self.reduction_features
                optimizer.orig_features = self.orig_features
                # 重要：先设置搜索方法
                optimizer.set_search_method(self.search_method)
                
                # 设置降维器
                if mode == 'pca' and PCAReducer:
                    optimizer.reducer = PCAReducer(self.reduction_features)
                    print(f"  使用PCA降维")
                elif mode == 'pls' and PLSReducer:
                    optimizer.reducer = PLSReducer(self.reduction_features)
                    print(f"  使用PLS降维")
                elif mode == 'svd' and SVDReducer:
                    optimizer.reducer = SVDReducer(self.reduction_features, svd_type='svd')
                    print(f"  使用标准SVD降维")
                elif mode == 'tsvd' and SVDReducer:
                    optimizer.reducer = SVDReducer(self.reduction_features, svd_type='tsvd')
                    print(f"  使用截断SVD降维")
                else:
                    optimizer.reducer = None
                    print(f"  不使用降维")
                
                # 重要：后初始化配置（这会初始化贝叶斯优化器）
                optimizer.initialize_model_configs(
                    tree_i=self.model_configs.get('xgb', {}).get('i', 2) if self.model_configs else 2,
                    tree_j=self.model_configs.get('xgb', {}).get('j', 5) if self.model_configs else 5,
                    line_i=self.model_configs.get('ridge', {}).get('i', 2) if self.model_configs else 2,
                    line_j=self.model_configs.get('ridge', {}).get('j', 5) if self.model_configs else 5,
                    fnn_i=self.model_configs.get('fnn', {}).get('i', 2) if self.model_configs else 2,
                    fnn_j=self.model_configs.get('fnn', {}).get('j', 3) if self.model_configs else 3,
                    reduction_type=mode if mode != 'none' else 'none',
                    model_list=model_list
                )
                
                # 运行优化
                available_models = list(optimizer.model_configs.keys())
                models_to_use = [m for m in model_list if m in available_models]
                
                if not models_to_use:
                    print(f"警告: {mode}模式没有可用的模型，跳过")
                    continue
                
                results, prop_results, best_model_info = optimizer.optimize_all_models(
                    temp_dir,  # output_dir
                    training_params_dir,  # training_params_dir
                    X.copy(), 
                    y, 
                    prop, 
                    models_to_use,
                    n_iterations, 
                    n_folds,
                    additional_save_path
                )
                
                # 保存结果
                mode_results[mode] = {
                    'results': results,
                    'best_model_info': best_model_info,
                    'temp_dir': temp_dir
                }
                
                if best_model_info:
                    print(f"{mode.upper()} 模式最佳test_score: {best_model_info.get('test_score', 'N/A'):.4f}")
                else:
                    print(f"{mode.upper()} 模式没有找到最佳模型")
                    
            except Exception as e:
                print(f"{mode.upper()} 模式运行失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 比较结果，选择最佳模式
        best_mode = None
        best_test_score = -float('inf')
        best_model_info = None
        
        for mode, result in mode_results.items():
            if result.get('best_model_info'):
                test_score = result['best_model_info'].get('test_score', -float('inf'))
                if test_score > best_test_score:
                    best_test_score = test_score
                    best_mode = mode
                    best_model_info = result['best_model_info']
        
        # 在这里直接处理summary文件，从temp目录读取并合并
        self._merge_summary_files(output_dir, mode_results, prop)
        
        if best_mode:
            mode_name = "CUSTOM" if custom_modes else "PLUS"
            print(f"\n{'='*80}")
            print(f"{mode_name}模式结果: 最佳模式为 {best_mode.upper()}")
            print(f"最佳test_score: {best_test_score:.4f}")
            print(f"{'='*80}")

            safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")
            final_perform_json = f"{output_dir}/best_model_perform.json"
            merged_perform = {}
            if os.path.exists(final_perform_json):
                with open(final_perform_json, 'r', encoding='utf-8') as f:
                    merged_perform = json.load(f)
            for mode, result in mode_results.items():
                temp_dir = result.get('temp_dir', '')
                temp_json = f"{temp_dir}/best_model_perform.json" if temp_dir else None
                if temp_json and os.path.exists(temp_json):
                    with open(temp_json, 'r', encoding='utf-8') as f:
                        mode_data = json.load(f)
                    merged_perform.update(mode_data)
            with open(final_perform_json, 'w', encoding='utf-8') as f:
                json.dump(merged_perform, f, ensure_ascii=False, indent=2)
            print(f"模型性能数据已合并保存到 '{final_perform_json}'")

            # 复制最佳模型到最终目录
            self._copy_best_model(mode_results[best_mode]['temp_dir'], prop, best_mode)
            
            # 清理临时目录
            self._cleanup_temp_dirs(mode_results, keep_best=True)
            
            # 保存模式比较结果
            self._save_plus_comparison(output_dir, mode_results, prop, best_mode, custom_modes)
            
            return {
                'best_mode': best_mode,
                'best_test_score': best_test_score,
                'best_model_info': best_model_info,
                'all_mode_results': mode_results
            }
        else:
            print(f"模式比较: 没有找到有效的最佳模型")
            # 清理所有临时目录
            self._cleanup_temp_dirs(mode_results, keep_best=False)

            return {}

    def _merge_summary_files(self, output_dir: str, mode_results: Dict, prop: str):
        """从temp目录读取summary文件并合并到最终JSON文件"""
        print(f"\n合并降维汇总文件 - 性质: {prop}")
        
        for mode, result in mode_results.items():
            temp_dir = result.get('temp_dir', '')
            if not temp_dir or not os.path.exists(temp_dir):
                continue
                
            summary_file_patterns = [
                f"{mode}_summary.json",
                f"{mode}_summary.csv",
                f"summary.json",
                f"summary.csv"
            ]
            
            summary_path = None
            for pattern in summary_file_patterns:
                temp_path = os.path.join(temp_dir, pattern)
                if os.path.exists(temp_path):
                    summary_path = temp_path
                    break
            
            if not summary_path:
                continue
                
            try:
                if summary_path.endswith('.json'):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if not isinstance(data, (dict, list)):
                        print(f"  警告: {mode.upper()} 汇总文件格式异常，跳过")
                        continue
                    if isinstance(data, dict):
                        data = [data]
                elif summary_path.endswith('.csv'):
                    df = pd.read_csv(summary_path)
                    data = df.to_dict(orient='records')
                else:
                    continue
                
                target_file = os.path.join(output_dir, f"{mode}_summary.json")
                existing_data = []
                if os.path.exists(target_file):
                    with open(target_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                
                for record in data:
                    record['性质'] = prop
                
                if isinstance(existing_data, dict):
                    for record in data:
                        prop_name = record.get('性质', prop)
                        existing_data[prop_name] = record
                else:
                    existing_data.extend(data)
                
                with open(target_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                print(f"  已更新 {mode.upper()} 汇总文件: {target_file}")
                    
            except Exception as e:
                print(f"  处理 {mode.upper()} 汇总文件失败: {e}")
    
    def _copy_best_model(self, source_dir: str, prop: str, mode: str):
        """复制最佳模型到最终目录"""
        safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")
        
        # 创建最终目录
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)
        
        # 复制文件
        files_to_copy = [
            f"best_model_{safe_prop}.pkl",
            f"best_model_{safe_prop}.pth", 
            f"best_features_{safe_prop}.json",
            f"best_params_{safe_prop}.json",
            f"best_model_info_{safe_prop}.json"
        ]
        
        for file_name in files_to_copy:
            source_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(self.best_model_dir, file_name)
            
            if os.path.exists(source_path):
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"  复制 {file_name} 到最终目录")
                except Exception as e:
                    print(f"  复制 {file_name} 失败: {e}")
    

    def _cleanup_temp_dirs(self, mode_results: Dict, keep_best: bool = False):
        """清理临时目录"""
        # 确定最佳模式
        best_mode = None
        if keep_best:
            best_mode = self._get_best_mode_from_results(mode_results)
        
        for mode, result in mode_results.items():
            temp_dir = result.get('temp_dir', '')
            if os.path.exists(temp_dir):
                try:
                    if keep_best and mode == best_mode:
                        print(f"保留最佳模式目录: {temp_dir}")
                    else:
                        shutil.rmtree(temp_dir)
                        print(f"清理临时目录: {temp_dir}")
                except Exception as e:
                    print(f"清理目录 {temp_dir} 失败: {e}")

    def _get_best_mode_from_results(self, mode_results: Dict) -> str:
        """从模式结果中获取最佳模式"""
        best_mode = None
        best_test_score = -float('inf')
        
        for mode, result in mode_results.items():
            if result.get('best_model_info'):
                test_score = result['best_model_info'].get('test_score', -float('inf'))
                if test_score > best_test_score:
                    best_test_score = test_score
                    best_mode = mode
        
        return best_mode
    
    def _save_plus_comparison(self, output_dir, mode_results: Dict, prop: str, best_mode: str, custom_modes: Optional[List[str]] = None):
        """保存模式比较结果"""
        try:
            comparison_data = []
            
            for mode, result in mode_results.items():
                if result.get('best_model_info'):
                    info = result['best_model_info']
                    comparison_data.append({
                        '降维模式': mode.upper(),
                        'test_score': info.get('test_score', 0),
                        '训练集R²': info.get('train_r2', 0),
                        '特征数量': info.get('num_features', 0),
                        '最佳模型': info.get('best_model', 'N/A'),
                        'orig_features': info.get('orig_features', 0),
                        'reduction_features': info.get('reduction_features', 0),
                        '是否最优': '是' if mode == best_mode else '否'
                    })
            
            if comparison_data:
                comparison_data.sort(key=lambda x: x['test_score'], reverse=True)
                
                if custom_modes:
                    json_file = f"{output_dir}/custom_mode_comparison.json"
                else:
                    json_file = f"{output_dir}/plus_mode_comparison.json"
                
                existing_data = []
                if os.path.exists(json_file):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                
                new_entry = {
                    '性质': prop,
                    '比较结果': comparison_data
                }
                existing_data.append(new_entry)
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
                print(f"模式比较结果已保存到 {json_file}")
                
                self._save_plus_summary(output_dir, prop, comparison_data, best_mode, custom_modes)
        except Exception as e:
            print(f"保存模式比较结果失败: {e}")
    
    def _save_plus_summary(self, output_dir, prop: str, comparison_data: List[Dict], best_mode: str, custom_modes: Optional[List[str]] = None):
        """保存模式汇总"""
        try:
            if custom_modes:
                summary_file = f"{output_dir}/custom_mode_summary.json"
                mode_type = "CUSTOM"
            else:
                summary_file = f"{output_dir}/plus_mode_summary.json"
                mode_type = "PLUS"
            
            best_info = next((item for item in comparison_data if item['降维模式'] == best_mode.upper()), None)
            
            if best_info:
                summary_row = {
                    '性质': prop,
                    f'最佳降维模式({mode_type})': best_mode.upper(),
                    '最佳test_score': best_info['test_score'],
                    '训练集R²': best_info['训练集R²'],
                    '特征数量': best_info['特征数量'],
                    'orig_features': best_info['orig_features'],
                    'reduction_features': best_info['reduction_features'],
                    '最佳模型': best_info['最佳模型']
                }
                
                existing_data = []
                if os.path.exists(summary_file):
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                existing_data.append(summary_row)
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                print(f"{mode_type}模式汇总已更新到 {summary_file}")
        except Exception as e:
            print(f"保存{mode_type}模式汇总失败: {e}")
    
    def process_multiple_properties(self, output_dir, training_params_dir: str, property_indices: List[int],
                                model_list: List[str], n_iterations: int = 10,
                                n_folds: int = 5, additional_save_path: Optional[str] = None,
                                df_features: pd.DataFrame = None,
                                df_targets: pd.DataFrame = None) -> Tuple[Dict, Dict]:
        """按目标分别剔除空值、划分数据集、标准化并训练模型。"""
        for idx in property_indices:
            if not 0 <= idx < len(self.property_names):
                raise ValueError(
                    f"property_index必须是0到{len(self.property_names) - 1}之间的整数，错误的索引: {idx}"
                )
        if df_features is None or df_targets is None:
            raise ValueError("必须提供 df_features 和 df_targets")

        df_features_full = df_features.copy()
        df_targets_full = df_targets.copy()
        target_properties = [self.property_names[i] for i in property_indices]
        target_properties = [
            prop for prop in target_properties
            if prop in df_targets_full.columns and not df_targets_full[prop].isna().all()
        ]
        skipped = [self.property_names[i] for i in property_indices if self.property_names[i] not in target_properties]
        for prop in skipped:
            print(f"警告: 目标 '{prop}' 不存在或全部为空，已跳过")

        print(
            f"成功加载完整数据集: Features形状={df_features_full.shape}, "
            f"Targets形状={df_targets_full.shape}"
        )
        print(
            f"将按目标独立划分训练/测试集 "
            f"(test_size={self.test_size}, random_state={self.random_state})"
        )

        all_results = {}
        best_models_info = {}
        id_col = df_features_full.columns[0]

        for prop in target_properties:
            print(f"\n{'#' * 100}")
            print(f"开始处理目标: {prop}")
            print(f"{'#' * 100}")

            X_train_scaled, y_train, X_test_scaled, y_test = split_data_per_property(
                df_features=df_features_full,
                df_targets=df_targets_full,
                property_name=prop,
                standardization_params_dir=training_params_dir,
                test_size=self.test_size,
                random_state=self.random_state,
                standardize=True,
            )
            self._per_property_test_data[prop] = {
                "test_ids": X_test_scaled[id_col].tolist() if len(X_test_scaled) else [],
                "X_test_scaled": X_test_scaled,
                "y_test": y_test,
            }

            X = X_train_scaled.set_index(id_col)
            y = y_train.copy()
            print(f"数据形状: X_train={X.shape}, y_train={y.shape}")

            results = {}
            if self.plus_mode:
                mode_result = self.run_plus_mode(
                    output_dir, training_params_dir, X, y, prop, model_list,
                    n_iterations, n_folds,
                    additional_save_path=additional_save_path,
                )
                if mode_result:
                    all_results[prop] = mode_result["all_mode_results"]
                    best_models_info[prop] = mode_result["best_model_info"]
            elif self.custom_mode:
                mode_result = self.run_plus_mode(
                    output_dir, training_params_dir, X, y, prop, model_list,
                    n_iterations, n_folds,
                    custom_modes=self.mode_list,
                    additional_save_path=additional_save_path,
                )
                if mode_result:
                    all_results[prop] = mode_result["all_mode_results"]
                    best_models_info[prop] = mode_result["best_model_info"]
            else:
                results, prop_results, best_model_info = self.optimize_all_models(
                    output_dir, training_params_dir, X, y, prop, model_list,
                    n_iterations, n_folds, additional_save_path,
                )
                all_results.update(prop_results)
                if best_model_info:
                    best_models_info[prop] = best_model_info
                self.print_final_results(results)

        return all_results, best_models_info

    def print_final_results(self, results: Dict):
        """打印最终结果"""
        if not results:
            return
            
        print(f"\n{'='*100}")
        print("模型降维、优化和最终训练结果汇总")
        print(f"{'='*100}")
        
        sorted_models = sorted(results.items(), key=lambda x: x[1]['test_score'], reverse=True)
        
        for rank, (model_type, result) in enumerate(sorted_models, 1):
            print(f"\n{rank}. 模型: {model_type}")
            print(f"   最终测试因子: {result['test_score']:.4f}")
            print(f"   特征数量: {result['num_features']}")
            print(f"   交叉验证平均R2: {result['avg_r2']:.4f}")
            print(f"   交叉验证正R2占比: {result['pos_rate']:.4f}")
            print(f"   交叉验证最大R2: {result['max_r2']:.4f}")
            print(f"   交叉验证平均相对误差: {result['perc_error']:.4f}")
            print(f"   交叉验证RMSE相对误差: {result.get('percentage_error_rmse', 'N/A'):.4f}")
            print(f"   交叉验证MAPE: {result.get('average_mape_test', 'N/A'):.4f}")
            
            if 'train_r2' in result:
                print(f"   最终训练集R2: {result['train_r2']:.4f}")
                print(f"   最终训练集RMSE: {result['train_rmse']:.4f}")
                print(f"   最终训练集MAE: {result['train_mae']:.4f}")
                print(f"   最终训练集平均相对误差: {result['percentage_error']:.4f}")
    
    def main(self, output_dir, training_params_dir: str, property_names: List[str], judge: int,
            property_indices: List[int], model_list: List[str],
            best_model_dir: str, n_iterations: int = 5, n_folds: int = 5, reduction_type: str = 'none',
            reduction_features: int = 50, orig_features: int = 0, 
            mode_list: Optional[List[str]] = None,
            clean_directory: bool = True,  
            search_method: str = 'random',
            additional_save_path: Optional[str] = None,
            train_features_df: pd.DataFrame = None,
            train_targets_df: pd.DataFrame = None,
            test_size: float = 0.2,
            random_state: int = 42,
            max_no_improve_rounds: int = 0,
            **kwargs) -> Tuple[Dict, pd.DataFrame, Dict]:
        """主函数"""
        self.property_names = property_names
        self.judge = judge
        self.max_no_improve_rounds = max_no_improve_rounds
        self.best_model_dir = best_model_dir
        self.reduction_features = reduction_features
        self.orig_features = orig_features  # 设置orig_features参数
        self.test_size = test_size
        self.random_state = random_state

        self.set_search_method(search_method)

        reduction_type_for_config = reduction_type

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(training_params_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)

        # 检查model_list是否包含特殊模式
        special_modes = ['plus', 'custom']
        for special_mode in special_modes:
            if special_mode in model_list:
                raise ValueError(f"model_list不能包含特殊模式 '{special_mode}'，请从model_list中移除'{special_mode}'")

        if clean_directory:
            print(f"\n{'='*80}")
            print("开始清理模型目录...")
            target_properties = [property_names[i] for i in property_indices]
            deleted, kept = self.clean_model_directory(output_dir, best_model_dir, target_properties, model_list)
            print("目录清理完成！")
            print(f"{'='*80}\n")

            # 项点更新场景下保留 best_model_perform.json（其他项点的评估记录），
            # 其内部 _save_perform_json 会按性质名合并更新，不清除
            result_files = [
                f"{output_dir}/plus_mode_comparison.json", 
                f"{output_dir}/custom_mode_comparison.json",  
                f"{output_dir}/plus_mode_summary.json",
                f"{output_dir}/custom_mode_summary.json",  
                f"{output_dir}/pca_summary.json",
                f"{output_dir}/pls_summary.json",
                f"{output_dir}/svd_summary.json",    
                f"{output_dir}/tsvd_summary.json",   
                f"{output_dir}/train_conclusion.json",
            ]
            for result_file in result_files:
                if os.path.exists(result_file):
                    os.remove(result_file)
                    print(f"  已清除旧文件: {result_file}")
            
            print(f"{'='*80}\n")

        # 打印orig_features参数信息
        if orig_features > 0:
            print(f"\norig_features参数: {orig_features}")

        # 设置神经网络后端
        if 'nn_backend' in kwargs:
            self.set_neural_network_backend(kwargs['nn_backend'], kwargs.get('nn_config'))
        
        # 设置降维模式
        if reduction_type == 'plus':
            self.plus_mode = True
            self.custom_mode = False
            print(f"\n使用PLUS模式: 将比较pca、pls、svd三种降维方式")
            print(f"降维特征数: {reduction_features}")
            
            # 对于PLUS模式，我们不需要设置具体的reducer
            self.reducer = None
            reduction_type_for_config = 'none'
        elif reduction_type == 'custom':
            self.custom_mode = True
            self.plus_mode = False
            self.mode_list = mode_list if mode_list else ['none']  # 使用传入的mode_list
            
            # 验证mode_list中的模式
            valid_modes = ['none', 'pca', 'pls', 'svd', 'tsvd']
            invalid_modes = [mode for mode in self.mode_list if mode not in valid_modes]
            if invalid_modes:
                raise ValueError(f"mode_list包含无效的模式: {invalid_modes}。有效模式为: {valid_modes}")
            
            print(f"\n使用CUSTOM模式: 将比较 {len(self.mode_list)} 种降维方式")
            print(f"模式列表: {', '.join(self.mode_list)}")
            print(f"降维特征数: {reduction_features}")
            
            # 对于CUSTOM模式，我们不需要设置具体的reducer
            self.reducer = None
            reduction_type_for_config = 'none'
        elif reduction_type == 'svd' and SVDReducer:
            self.reducer = SVDReducer(reduction_features, svd_type='svd')
            print(f"使用标准SVD降维，特征数: {reduction_features}")
            self.plus_mode = False
            self.custom_mode = False
            reduction_type_for_config = 'svd'
        elif reduction_type == 'tsvd' and SVDReducer:
            self.reducer = SVDReducer(reduction_features, svd_type='tsvd')
            print(f"使用截断SVD(TruncatedSVD)降维，特征数: {reduction_features}")
            self.plus_mode = False
            self.custom_mode = False
            reduction_type_for_config = 'tsvd'
        elif reduction_type == 'pca' and PCAReducer:
            self.reducer = PCAReducer(reduction_features)
            print(f"使用PCA降维，特征数: {reduction_features}")
            self.plus_mode = False
            self.custom_mode = False
        elif reduction_type == 'pls' and PLSReducer:
            self.reducer = PLSReducer(reduction_features)
            print(f"使用PLS降维，特征数: {reduction_features}")
            self.plus_mode = False
            self.custom_mode = False
        else:
            self.reducer = None
            self.plus_mode = False
            self.custom_mode = False
            print("不使用降维")
            
        # 初始化模型配置
        self.initialize_model_configs(reduction_type=reduction_type_for_config, model_list=model_list)
        
        # 处理多个性质
        all_results, best_models_info = self.process_multiple_properties(
            output_dir, training_params_dir, property_indices, 
            model_list, n_iterations, n_folds, additional_save_path,
            df_features=train_features_df, df_targets=train_targets_df,
        )
        
        print(f"\n{'='*100}")
        print("模型优化完成！")
        
        if best_models_info:
            print(f"\n各性质最佳模型汇总:")
            for prop, info in best_models_info.items():
                if self.plus_mode or self.custom_mode:
                    reduction_info = f", 最佳降维模式: {info.get('reduction_type', 'N/A').upper()}"
                elif self.reducer and 'reduction_ratio' in info:
                    reduction_info = f", {self.reducer.reduction_type}比例: {info['reduction_ratio']:.2f}"
                else:
                    reduction_info = ""
                
                # 添加orig_features信息
                # orig_features_info = f", 使用前{info.get('orig_features', 0)}列特征" if info.get('orig_features', 0) > 0 else ""
                if info.get('best_model', 'N/A') in ['fnn', 'deep_fnn', 'simple_fnn', 'resnet']:
                    frame = self.nn_backend
                else:
                    frame = 'sklearn'
                # print(f" 性质: {prop}, 最佳模型: {info.get('best_model', 'N/A')}, 测试因子: {info.get('test_score', 'N/A'):.4f}, 训练集R²: {info.get('train_r2', 'N/A'):.4f}{reduction_info}, 使用框架: {frame}")
                # print(f'''  {prop}: {info.get('best_model', 'N/A')} (test因子: {info.get('test_score', 'N/A'):.4f if isinstance(info.get('test_score'), (int, float)) else info.get('test_score', 'N/A')}, 训练集R²: {info.get('train_r2', 'N/A'):.4f if isinstance(info.get('train_r2'), (int, float)) else info.get('train_r2', 'N/A')}), 
                                    # 搜索方法：{search_method}，最佳降维模式: {info.get('reduction_type', 'N/A').upper()}, 使用框架：{frame}''')             
                record = {
                    '性质': prop,
                    '最佳模型': info.get('best_model', 'N/A'),
                    '测试因子': info.get('test_score', 'N/A'),
                    '训练集R²': info.get('train_r2', 'N/A'),
                    '搜索方法': search_method,
                    '最佳降维模式': info.get('reduction_type', 'N/A').upper(),
                    '使用框架': frame
                }

                file_path = Path(output_dir) / 'train_conclusion.json'
                existing_data = []
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                existing_data.append(record)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        # 返回空DataFrame以保持接口兼容
        test_performance_df = pd.DataFrame()

        if os.path.exists("catboost_info"):
            try:
                shutil.rmtree("catboost_info")
                print("已自动删除 catboost_info 目录")
            except Exception as e:
                print(f"清理 catboost_info 失败: {e}")

        for root, dirs, _ in os.walk('.', topdown=False):
            for d in dirs:
                if d.lower().startswith(f'temp'):
                    shutil.rmtree(os.path.join(root, d))

        
        return all_results, test_performance_df, best_models_info
