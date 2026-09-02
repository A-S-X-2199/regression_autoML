import numpy as np
import pandas as pd
import os
from typing import Dict, List, Any, Tuple, Optional, Union

# 设置环境变量，避免joblib创建子进程失败
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['JOBLIB_MULTIPROCESSING'] = '0'

# 尝试导入skopt，如果不可用则提供备选方案
try:
    from skopt import Optimizer
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("警告: scikit-optimize 库未安装，贝叶斯优化功能将不可用")
    print("请使用: pip install scikit-optimize")

# 尝试导入sklearn的kernel
try:
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
    SKLEARN_KERNELS_AVAILABLE = True
except ImportError:
    SKLEARN_KERNELS_AVAILABLE = False


class BayesianParamTransformer:
    """贝叶斯优化参数转换器"""
    
    def __init__(self):
        """初始化参数转换器"""
        self.param_info = {}  # 存储参数信息
        self.encoded_spaces = {}  # 存储编码后的参数空间
        self.kernel_mapping = {}  # 存储kernel对象的映射
        
    def encode_param_space(self, model_name: str, param_space: Dict) -> Dict:
        """将参数空间编码为贝叶斯优化友好的格式"""
        encoded_space = {}
        param_info = {}
        
        for param_name, param_values in param_space.items():
            # 确保param_values是列表
            if not isinstance(param_values, list):
                param_values = [param_values]
                
            # 跳过空参数
            if not param_values:
                continue
            
            # 特别处理kernel参数
            if param_name == 'kernel' and SKLEARN_KERNELS_AVAILABLE:
                # 对于kernel参数，我们将其转换为索引表示
                kernel_indices = list(range(len(param_values)))
                param_info[param_name] = {
                    'type': 'kernel',
                    'values': param_values,
                    'indices': kernel_indices
                }
                encoded_space[param_name] = kernel_indices
                continue

            elif param_name == 'hidden_layer_sizes':
                # 从给定的元组列表中提取信息
                all_tuples = [tuple(t) for t in param_values if isinstance(t, tuple) or isinstance(t, list)]
                
                if not all_tuples:
                    # 如果没有有效的元组，使用默认值
                    encoded_space[param_name] = param_values
                    param_info[param_name] = {
                        'type': 'categorical',
                        'values': param_values
                    }
                    continue
                
                # 计算层数范围
                layer_counts = [len(t) for t in all_tuples]
                max_layers = max(layer_counts)
                min_layers = min(layer_counts)
                
                # 计算第一层神经元数范围
                first_layer_sizes = [t[0] for t in all_tuples if len(t) > 0]
                max_first = max(first_layer_sizes)
                min_first = min(first_layer_sizes)
                
                # 检测是否为等维层架构（如ResNet：所有层神经元数相同）
                is_uniform = all(
                    len(set(t)) == 1 for t in all_tuples if len(t) > 1
                ) and len(all_tuples) > 0
                
                if is_uniform:
                    # 等维架构：各层维度必须相同，无需decay_rate维度
                    # 只优化层数和每层神经元数
                    if min_layers == max_layers:
                        max_layers = min_layers + 1
                    if min_first == max_first:
                        max_first = min_first * 2
                    
                    param_info[param_name] = {
                        'type': 'nn_architecture_uniform',
                        'min_layers': min_layers,
                        'max_layers': max_layers,
                        'min_first': min_first,
                        'max_first': max_first,
                        'original_tuples': all_tuples
                    }
                    
                    encoded_space[f'{param_name}_num_layers'] = (min_layers, max_layers)
                    encoded_space[f'{param_name}_first_layer'] = (min_first, max_first)
                    # 不添加 decay_rate 维度，_fix_neural_network_params 会检测并生成均匀层
                    continue
                
                # 计算衰减率范围（通过分析相邻层的比例）
                decay_rates = []
                for t in all_tuples:
                    if len(t) > 1:
                        for i in range(1, len(t)):
                            if t[i-1] > 0:  # 避免除以零
                                ratio = t[i] / t[i-1]
                                decay_rates.append(ratio)
                
                # 设置衰减率范围
                if decay_rates:
                    min_decay = min(decay_rates)
                    max_decay = max(decay_rates)
                else:
                    # 如果没有多层结构，使用默认衰减率
                    min_decay = 0.25
                    max_decay = 1.0
                
                # 确保范围合理
                min_decay = max(0.1, min_decay)  # 最小衰减率不低于0.1
                max_decay = min(1.0, max_decay)  # 最大衰减率不超过1.0
                
                # 重要修复：确保上下界不相等
                if min_decay == max_decay:
                    # 如果相等，进行微调
                    if min_decay > 0.1:
                        min_decay = max(0.1, min_decay * 0.9)  # 减少10%
                    else:
                        max_decay = min(1.0, max_decay * 1.1)  # 增加10%
                
                # 同样处理层数和第一层神经元数范围
                if min_layers == max_layers:
                    max_layers = min_layers + 1  # 至少增加一层
                
                if min_first == max_first:
                    max_first = min_first * 2  # 加倍
                
                # 存储架构参数信息
                param_info[param_name] = {
                    'type': 'nn_architecture',
                    'min_layers': min_layers,
                    'max_layers': max_layers,
                    'min_first': min_first,
                    'max_first': max_first,
                    'min_decay': min_decay,
                    'max_decay': max_decay,
                    'original_tuples': all_tuples  # 保留原始结构用于参考
                }
                
                # 编码为三个独立的优化维度
                encoded_space[f'{param_name}_num_layers'] = (min_layers, max_layers)
                encoded_space[f'{param_name}_first_layer'] = (min_first, max_first)
                encoded_space[f'{param_name}_decay_rate'] = (min_decay, max_decay)
                
                continue
                       
            # 处理单个固定值的情况
            elif len(param_values) == 1:
                param_info[param_name] = {
                    'type': 'fixed',
                    'value': param_values[0]
                }
                encoded_space[param_name] = param_values[0]
                continue
                
            # 判断参数类型
            param_type = self._detect_param_type(param_values)
            
            if param_type == 'numeric':
                # 数值型参数
                numeric_values = self._extract_numeric_values(param_values)
                if numeric_values:
                    min_val, max_val = self._get_numeric_bounds(numeric_values)
                    
                    # 检查是否是整数型
                    all_integers = self._are_all_integers(numeric_values)
                    
                    if all_integers:
                        param_info[param_name] = {
                            'type': 'integer',
                            'min': int(min_val),
                            'max': int(max_val),
                            'original_values': param_values
                        }
                    else:
                        param_info[param_name] = {
                            'type': 'numeric',
                            'min': min_val,
                            'max': max_val,
                            'original_values': param_values
                        }
                    # 对于数值型参数，返回连续范围
                    encoded_space[param_name] = (min_val, max_val)
                else:
                    # 转为分类参数
                    param_info[param_name] = {
                        'type': 'categorical',
                        'values': param_values
                    }
                    encoded_space[param_name] = param_values  # 存储原始值列表
                    
            elif param_type == 'categorical':
                # 分类参数 - 存储原始值列表，而不是Categorical对象
                param_info[param_name] = {
                    'type': 'categorical',
                    'values': param_values
                }
                encoded_space[param_name] = param_values  # 存储原始值列表
                
            elif param_type == 'mixed':
                # 分离数值和非数值部分
                numeric_vals = []
                categorical_vals = []
                
                for val in param_values:
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        numeric_vals.append(float(val))
                    else:
                        categorical_vals.append(val)
                
                # 情况A: 主要是数值，包含少量特殊字符串（如 'auto'）
                if len(numeric_vals) > len(categorical_vals):
                    param_info[param_name] = {
                        'type': 'mixed_with_numeric',
                        'numeric_range': (min(numeric_vals), max(numeric_vals)),
                        'special_options': categorical_vals, # 如 ['auto']
                        'all_values': param_values
                    }
                    # 编码为连续范围 + 特殊选项标志
                    # 可以设计为两个维度：一个连续值，一个是否启用特殊选项的布尔值
                    encoded_space[param_name] = (min(numeric_vals), max(numeric_vals))
                    encoded_space[f'{param_name}_use_special'] = [False, True] if categorical_vals else [False]
                
                # 情况B: 数值和分类数量相当，或主要是分类
                else:
                    # 退化为当前方案：当作纯分类处理
                    param_info[param_name] = {
                        'type': 'categorical',
                        'values': param_values
                    }
                    encoded_space[param_name] = param_values
                continue
        
        self.param_info[model_name] = param_info
        self.encoded_spaces[model_name] = encoded_space
        return encoded_space
    
    def get_param_dimensions(self, model_name: str) -> List:
        """获取参数的skopt维度列表"""
        if model_name not in self.param_info:
            return []
        
        dimensions = []
        param_info = self.param_info[model_name]
        
        for param_name, info in param_info.items():
            if info['type'] in ('nn_architecture', 'nn_architecture_uniform'):
                # 添加层数维度（整数）
                min_layers = info['min_layers']
                max_layers = info['max_layers']
                if min_layers < max_layers:
                    dimensions.append(
                        Integer(int(min_layers), int(max_layers), 
                            name=f'{param_name}_num_layers')
                    )
                else:
                    print(f"注意: {param_name}_num_layers 上下界相等 ({min_layers})，将作为固定值处理")
                
                # 添加第一层神经元数维度（整数）
                min_first = info['min_first']
                max_first = info['max_first']
                if min_first < max_first:
                    dimensions.append(
                        Integer(int(min_first), int(max_first), 
                            name=f'{param_name}_first_layer')
                    )
                else:
                    print(f"注意: {param_name}_first_layer 上下界相等 ({min_first})，将作为固定值处理")
                
                # 等维架构（ResNet）不需要 decay_rate 维度
                if info['type'] == 'nn_architecture_uniform':
                    continue
                
                # 添加衰减率维度（实数）
                min_decay = info['min_decay']
                max_decay = info['max_decay']
                if min_decay < max_decay:
                    dimensions.append(
                        Real(min_decay, max_decay, 
                            name=f'{param_name}_decay_rate')
                    )
                else:
                    print(f"注意: {param_name}_decay_rate 上下界相等 ({min_decay})，将作为固定值处理")
                continue
            
            elif info['type'] == 'fixed':
                # 固定参数，不添加到优化维度中
                continue
            
            elif info['type'] == 'kernel':
                # kernel参数：使用索引作为分类参数
                indices = info.get('indices', [])
                if indices:
                    dimensions.append(Categorical(indices, name=param_name))
            
            elif info['type'] == 'numeric':
                # 数值型参数
                if model_name in self.encoded_spaces and param_name in self.encoded_spaces[model_name]:
                    param_range = self.encoded_spaces[model_name][param_name]
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        min_val, max_val = param_range
                        # 确保范围有效
                        if min_val < max_val:
                            dimensions.append(Real(min_val, max_val, name=param_name))
                        else:
                            # 如果范围无效，调整范围
                            print(f"警告: {param_name} 范围无效 ({min_val}, {max_val})，调整为默认范围")
                            dimensions.append(Real(0.0, 1.0, name=param_name))
            
            elif info['type'] == 'integer':
                # 整数型参数
                if model_name in self.encoded_spaces and param_name in self.encoded_spaces[model_name]:
                    param_range = self.encoded_spaces[model_name][param_name]
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        min_val, max_val = param_range
                        # 确保范围有效
                        if min_val < max_val:
                            dimensions.append(Integer(int(min_val), int(max_val), name=param_name))
                        else:
                            # 如果范围无效，调整范围
                            print(f"警告: {param_name} 范围无效 ({min_val}, {max_val})，调整为默认范围")
                            dimensions.append(Integer(0, 10, name=param_name))
            
            elif info['type'] == 'categorical':
                # 分类参数 - 确保使用原始值列表，而不是Categorical对象
                if 'values' in info and isinstance(info['values'], list) and len(info['values']) > 1:
                    # 检查值是否可哈希
                    try:
                        hash(info['values'][0])
                        dimensions.append(Categorical(info['values'], name=param_name))
                    except (TypeError, IndexError):
                        # 如果值不可哈希，转换为字符串表示
                        str_values = [str(v) for v in info['values']]
                        dimensions.append(Categorical(str_values, name=param_name))
                        # 更新param_info中的值为字符串表示
                        info['values'] = str_values
        
        # 循环结束后返回所有维度
        return dimensions
    
    def get_fixed_params(self, model_name: str) -> Dict:
        """获取固定参数"""
        if model_name not in self.param_info:
            return {}
        
        fixed_params = {}
        param_info = self.param_info[model_name]
        
        for param_name, info in param_info.items():
            if info['type'] == 'fixed':
                fixed_params[param_name] = info['value']
        
        return fixed_params
    
    def decode_params(self, model_name: str, encoded_params: Dict) -> Dict:
        """将编码后的参数解码为原始格式"""
        if model_name not in self.param_info:
            return encoded_params
            
        decoded_params = {}
        param_info = self.param_info[model_name]
        min_neu = 5
        
        for param_name, encoded_value in encoded_params.items():
            if param_name not in param_info:
                decoded_params[param_name] = encoded_value
                continue
                
            info = param_info[param_name]

            if info['type'] in ('nn_architecture', 'nn_architecture_uniform'):
                # 从编码参数中提取维度的值
                num_layers_key = f'{param_name}_num_layers'
                first_layer_key = f'{param_name}_first_layer'
                decay_rate_key = f'{param_name}_decay_rate'
                
                # 获取优化后的值，如果不存在则使用默认值
                num_layers = int(np.round(float(encoded_params.get(
                    num_layers_key, 
                    (info['min_layers'] + info['max_layers']) / 2
                ))))
                
                first_layer = int(np.round(float(encoded_params.get(
                    first_layer_key,
                    (info['min_first'] + info['max_first']) / 2
                ))))
                
                # 确保值在合理范围内
                num_layers = int(np.clip(num_layers, info['min_layers'], info['max_layers']))
                first_layer = int(np.clip(first_layer, info['min_first'], info['max_first']))
                
                # 等维架构（ResNet）：所有层统一大小
                if info['type'] == 'nn_architecture_uniform':
                    decay_rate = 1.0
                else:
                    decay_rate = float(encoded_params.get(
                        decay_rate_key,
                        (info['min_decay'] + info['max_decay']) / 2
                    ))
                    decay_rate = float(np.clip(decay_rate, info['min_decay'], info['max_decay']))
                
                # 生成神经网络层结构元组
                layers = []
                current_size = first_layer
                
                for i in range(num_layers):
                    layer_size = int(np.round(current_size))
                    layer_size = max(min_neu, layer_size)
                    layers.append(layer_size)
                    current_size = current_size * decay_rate
                
                decoded_params[param_name] = tuple(layers)
                continue

            elif info['type'] == 'fixed':
                decoded_params[param_name] = info['value']
            elif info['type'] == 'kernel':
                # 特别处理kernel参数：根据索引获取实际的kernel对象
                kernel_values = info.get('values', [])
                if isinstance(encoded_value, int) and 0 <= encoded_value < len(kernel_values):
                    decoded_params[param_name] = kernel_values[encoded_value]
                elif isinstance(encoded_value, str) and encoded_value.isdigit():
                    idx = int(encoded_value)
                    if 0 <= idx < len(kernel_values):
                        decoded_params[param_name] = kernel_values[idx]
                    else:
                        # 默认使用第一个kernel
                        decoded_params[param_name] = kernel_values[0] if kernel_values else None
                else:
                    # 如果无法解析，使用第一个kernel
                    decoded_params[param_name] = kernel_values[0] if kernel_values else None
            elif info['type'] == 'numeric':
                # 确保数值在合理范围内
                min_val = info.get('min', -float('inf'))
                max_val = info.get('max', float('inf'))
                decoded_params[param_name] = float(np.clip(float(encoded_value), min_val, max_val))  # 转原生float
            elif info['type'] == 'integer':
                min_val = info.get('min', -np.inf)
                max_val = info.get('max', np.inf)
                int_value = int(np.round(float(encoded_value)))
                decoded_params[param_name] = int(np.clip(int_value, min_val, max_val))  # 转原生int
            elif info['type'] == 'categorical':
                # 对于分类参数，直接使用优化器给出的值
                decoded_params[param_name] = encoded_value
                # 验证值是否在允许的范围内
                if 'values' in info and encoded_value not in info['values']:
                    # 如果值不在列表中，选择第一个值
                    if info['values']:
                        decoded_params[param_name] = info['values'][0]
        
        return decoded_params
    
    def _detect_param_type(self, param_values: List) -> str:
        """检测参数类型"""
        # 检查是否包含非数值元素
        has_non_numeric = False
        has_numeric = False
        
        for val in param_values:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                has_numeric = True
            elif isinstance(val, str):
                # 检查字符串是否可转为数字
                if self._is_number_string(val):
                    has_numeric = True
                else:
                    has_non_numeric = True
            elif isinstance(val, tuple):
                # 包含元组，视为混合类型
                return 'mixed'
            elif self._is_kernel_object(val):
                # 如果是kernel对象，视为特殊类型
                return 'kernel'
            else:
                # 其他类型（如bool, None等）
                has_non_numeric = True
        
        if has_numeric and not has_non_numeric:
            # 纯数值型
            return 'numeric'
        elif has_numeric and has_non_numeric:
            # 混合类型
            return 'mixed'
        else:
            # 纯分类型
            return 'categorical'
    
    def _is_kernel_object(self, obj) -> bool:
        """检查对象是否为sklearn kernel对象"""
        if not SKLEARN_KERNELS_AVAILABLE:
            return False
        
        try:
            # 检查对象是否具有kernel对象的特征
            return hasattr(obj, '__class__') and hasattr(obj, '__repr__') and \
                   'kernel' in obj.__class__.__name__.lower()
        except:
            return False
    
    def _is_number_string(self, s: str) -> bool:
        """检查字符串是否可以转换为数字"""
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False
    
    def _are_all_integers(self, numeric_values: List[float]) -> bool:
        """检查所有数值是否都是整数"""
        return all((v.is_integer() if isinstance(v, float) else isinstance(v, int)) for v in numeric_values)
    
    def _extract_numeric_values(self, values: List) -> List[float]:
        """从参数值列表中提取数值型值"""
        numeric_values = []
        for val in values:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_values.append(float(val))
            elif isinstance(val, str):
                try:
                    numeric_values.append(float(val))
                except (ValueError, TypeError):
                    continue
        return numeric_values
    
    def _get_numeric_bounds(self, numeric_values: List[float]) -> Tuple[float, float]:
        """获取数值型参数的边界"""
        if not numeric_values:
            return 0.0, 1.0
            
        min_val = min(numeric_values)
        max_val = max(numeric_values)
        
        # 确保范围不为零
        if min_val == max_val:
            max_val = min_val + 1.0
        
        return min_val, max_val

class BayesianOptimizer:
    """贝叶斯优化器"""
    
    def __init__(self, base_estimator='gp', n_initial_ratio=0.3, 
                 acq_func='EI', acq_optimizer='auto', random_state=42):
        """
        初始化贝叶斯优化器
        
        参数:
        base_estimator: 基础估计器 ('gp', 'rf', 'et', 'gbm')
        n_initial_ratio: 初始随机采样点数比例 (0-1之间)
        acq_func: 采集函数 ('EI', 'PI', 'LCB')
        acq_optimizer: 采集优化器 ('auto', 'sampling', 'lbfgs')
        random_state: 随机种子
        """
        import os
        # 设置环境变量，避免joblib创建子进程失败
        os.environ['LOKY_MAX_CPU_COUNT'] = '1'
        
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize 库未安装，无法使用贝叶斯优化")
            
        self.base_estimator = base_estimator
        self.n_initial_ratio = n_initial_ratio
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer
        self.random_state = random_state
        
        self.transformer = BayesianParamTransformer()
        self.optimizers = {}
        self.best_params = {}
        self.best_score = {}
    
    def prepare_optimizer(self, model_name: str, param_space: Dict) -> bool:
        """为指定模型准备贝叶斯优化器配置（不创建Optimizer实例）"""
        try:
            # 编码参数空间
            encoded_space = self.transformer.encode_param_space(model_name, param_space)
            
            # 获取参数的skopt维度
            dimensions = self.transformer.get_param_dimensions(model_name)
            
            if not dimensions:
                print(f"模型 {model_name} 没有可优化的参数")
                return False
            
            # 获取固定参数
            fixed_params = self.transformer.get_fixed_params(model_name)
            
            # 调试信息：显示创建的维度
            print(f"模型 {model_name}: 创建了 {len(dimensions)} 个优化维度")
            for dim in dimensions:
                if hasattr(dim, 'low') and hasattr(dim, 'high'):
                    print(f"  - {dim.name}: ({dim.low}, {dim.high})")
                elif hasattr(dim, 'categories'):
                    print(f"  - {dim.name}: {dim.categories}")
            
            # 存储优化器配置信息（但不创建Optimizer实例）
            self.optimizers[model_name] = {
                'dimensions': dimensions,
                'param_names': [dim.name for dim in dimensions],
                'fixed_params': fixed_params,
                'param_space': param_space,
                'encoded_space': encoded_space
            }
            
            return True
            
        except Exception as e:
            print(f"准备模型 {model_name} 的贝叶斯优化器时出错: {e}")
            print(f"参数空间: {param_space}")
            import traceback
            traceback.print_exc()
            return False
        
    def optimize(self, model_name: str, objective_func, n_calls=20, verbose=False):
        """执行贝叶斯优化"""
        if model_name not in self.optimizers:
            raise ValueError(f"模型 {model_name} 的优化器未准备。请先调用 prepare_optimizer。")
        
        opt_info = self.optimizers[model_name]
        dimensions = opt_info['dimensions']
        fixed_params = opt_info['fixed_params']
        param_names = opt_info['param_names']
        
        # 根据 n_initial_ratio 计算初始点数
        n_initial_points = max(1, int(round(n_calls * self.n_initial_ratio)))
        n_initial_points = min(n_initial_points, n_calls - 1)  # 确保至少有一次非随机搜索
        
        if verbose:
            print(f"初始随机采样点数: {n_initial_points} (基于 n_initial_ratio={self.n_initial_ratio})")
        
        # 根据模型类型调整迭代次数
        model_list_low = [
            'linear', 'ridge', 'lasso', 'elasticnet', 'bayesian_ridge',
            'dt', 'linearsvr', 'knn', 'huber', 'poly',
        ]

        model_list_medium = [
            'svr', 'svr_rbf', 'rf', 'extra_trees', 'gbr', 'gbdt', 'hist_gbdt',
            'adaboost', 'gpr', 'xgb', 'lgbm', 'catboost',
        ]

        model_list_high = [
            'fnn', 'deep_fnn', 'simple_fnn', 'resnet',
        ]
        
        if model_name in model_list_medium:
            n_calls = round(n_calls * 0.7)
        elif model_name in model_list_high:
            n_calls = round(n_calls * 0.5)
        
        # 创建优化器（每次都新建，确保正确的初始点数）
        optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator=self.base_estimator,  # 用字符串 'gp'，skopt 内部自动归一化并处理兼容性
            n_initial_points=n_initial_points,
            acq_func=self.acq_func,
            acq_optimizer=self.acq_optimizer,
            random_state=self.random_state,
            n_jobs=1  # 设置为单线程，避免子进程创建问题
        )
        
        if verbose:
            print(f"开始贝叶斯优化 {model_name}，总迭代次数: {n_calls}")
            print(f"可优化参数: {param_names}")
            if fixed_params:
                print(f"固定参数: {fixed_params}")
        
        best_score = -float('inf')
        best_params = {}
        completed_iterations = 0
        
        for i in range(n_calls):
            try:
                # 获取下一个采样点
                try:
                    suggested = optimizer.ask()
                except RuntimeError as e:
                    if "Random evaluations exhausted" in str(e):
                        if verbose:
                            print(f"警告: 随机评估已用尽，使用随机参数")
                        # 生成随机参数
                        suggested = []
                        for dim in dimensions:
                            if isinstance(dim, Real):
                                suggested.append(np.random.uniform(dim.low, dim.high))
                            elif isinstance(dim, Integer):
                                suggested.append(np.random.randint(dim.low, dim.high))
                            elif isinstance(dim, Categorical):
                                suggested.append(np.random.choice(dim.categories))
                            else:
                                suggested.append(0)
                    else:
                        raise
                except AttributeError as e:
                    if "_next_x" in str(e):
                        if verbose:
                            print(f"警告: Optimizer._next_x 未就绪 ({e})，使用随机参数")
                        suggested = []
                        for dim in dimensions:
                            if isinstance(dim, Real):
                                suggested.append(np.random.uniform(dim.low, dim.high))
                            elif isinstance(dim, Integer):
                                suggested.append(np.random.randint(dim.low, dim.high))
                            elif isinstance(dim, Categorical):
                                suggested.append(np.random.choice(dim.categories))
                            else:
                                suggested.append(0)
                    else:
                        raise
                
                # 构建参数字典
                params_dict = dict(zip(param_names, suggested))
                
                # 合并固定参数
                full_params = {**params_dict, **fixed_params}
                
                # 解码参数（确保格式正确）
                decoded_params = self.transformer.decode_params(model_name, full_params)
                
                # 特别验证kernel参数
                if 'kernel' in decoded_params:
                    kernel_value = decoded_params['kernel']
                    # 确保kernel值不是字符串
                    if isinstance(kernel_value, str):
                        if verbose:
                            print(f"警告: kernel参数是字符串 '{kernel_value}'，尝试转换为kernel对象")
                        # 尝试从param_info中获取kernel对象
                        if model_name in self.transformer.param_info and \
                        'kernel' in self.transformer.param_info[model_name]:
                            kernel_info = self.transformer.param_info[model_name]['kernel']
                            if 'values' in kernel_info and isinstance(kernel_info['values'], list):
                                # 假设字符串是索引
                                try:
                                    idx = int(kernel_value)
                                    if 0 <= idx < len(kernel_info['values']):
                                        decoded_params['kernel'] = kernel_info['values'][idx]
                                except:
                                    # 使用第一个kernel
                                    decoded_params['kernel'] = kernel_info['values'][0] if kernel_info['values'] else None
                
                # 评估目标函数，并处理可能的无限值
                try:
                    score = objective_func(decoded_params)
                    
                    # 检查分数是否有效
                    if not np.isfinite(score):
                        if verbose:
                            print(f"警告: 迭代 {i+1} 返回了无效分数 {score}，使用默认值 -1e6")
                        score = -1e6  # 使用非常差但有限的值
                    
                except Exception as e:
                    if verbose:
                        print(f"目标函数在迭代 {i+1} 失败: {e}")
                    score = -1e6  # 使用非常差但有限的值
                
                # 告诉优化器结果（使用负分数，因为optimizer最小化目标）
                try:
                    optimizer.tell(suggested, -score)
                except Exception as e:
                    if verbose:
                        print(f"优化器更新失败，但继续执行: {e}")
                    # 继续执行，不更新优化器
                
                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_params = decoded_params.copy()
                
                completed_iterations += 1
                
                if verbose and (i + 1) % max(1, n_calls // 10) == 0:
                    print(f"  迭代 {i+1}/{n_calls}: 当前分数 = {score:.4f}, 最佳分数 = {best_score:.4f} \n 当前参数：{best_params}" )
                    
            except Exception as e:
                print(f"贝叶斯优化迭代 {i+1} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 关键修复：这里应该返回 best_score，而不是 _
        if completed_iterations == 0:
            print(f"警告: 模型 {model_name} 的所有迭代都失败了，返回默认参数")
            best_params = fixed_params.copy()
            best_score = -float('inf')

        best_params = self._ensure_python_types(best_params)
        best_params = self._clean_numpy_objects(best_params)

        # 保存最佳结果
        self.best_params[model_name] = best_params
        self.best_score[model_name] = best_score
        
        if verbose:
            print(f"贝叶斯优化完成，完成迭代: {completed_iterations}/{n_calls}，最佳分数: {best_score:.4f}")
        
        # 重要：返回 best_score
        return best_params, best_score


    def _ensure_python_types(self, params_dict):
        """确保参数字典中所有值为Python原生类型"""
        import numpy as np
        
        def convert(obj):
            if obj is None:
                return None
            elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert(v) for v in obj)
            else:
                return obj
        
        return convert(params_dict)

    def _clean_numpy_objects(self, obj):
        """递归清理NumPy对象"""
        import numpy as np
        
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                # 清理键
                if isinstance(k, np.generic):
                    k = k.item()
                # 清理值
                cleaned[k] = self._clean_numpy_objects(v)
            return cleaned
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._clean_numpy_objects(v) for v in obj)
        elif isinstance(obj, np.generic):  # 所有NumPy标量
            return obj.item()  # 转换为Python标量
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


    def get_best_params(self, model_name: str) -> Dict:
        """获取最佳参数"""
        return self.best_params.get(model_name, {})
    
    def get_best_score(self, model_name: str) -> float:
        """获取最佳分数"""
        return self.best_score.get(model_name, -float('inf'))


class BayesianHyperparamSearcher:
    """贝叶斯超参数搜索器（与主优化器集成）"""
    
    def __init__(self, model_optimizer):
        self.model_optimizer = model_optimizer
        if SKOPT_AVAILABLE:
            self.bayesian_optimizer = BayesianOptimizer(
                base_estimator='gp',
                n_initial_ratio=0.3, 
                acq_func='EI',
                random_state=42
            )
        else:
            self.bayesian_optimizer = None
            print("警告: scikit-optimize 未安装，贝叶斯优化不可用")
        
        # 初始化优化器
        self._initialize_bayesian_optimizers()
    
    def _initialize_bayesian_optimizers(self):
        """为所有模型初始化贝叶斯优化器"""
        if self.bayesian_optimizer is None:
            return  # 贝叶斯优化器不可用，静默返回
        
        if not hasattr(self.model_optimizer, 'model_configs') or not self.model_optimizer.model_configs:
            return  # 模型配置为空，静默返回
        
        success_count = 0
        for model_name, config in self.model_optimizer.model_configs.items():
            param_space = config['param_space']
            try:
                if not param_space:
                    continue
                    
                result = self.bayesian_optimizer.prepare_optimizer(model_name, param_space)
                if result:
                    success_count += 1
                else:
                    print(f"模型 {model_name} 的贝叶斯优化器准备失败")
            except Exception as e:
                # 静默处理错误
                print(f"模型 {model_name} 初始化贝叶斯优化器时出错: {e}")
        
        if success_count > 0:
            print(f"贝叶斯优化器初始化完成: {success_count}/{len(self.model_optimizer.model_configs)} 个模型成功")
            
        # 同步优化器配置
        if hasattr(self.model_optimizer, 'search_method') and self.model_optimizer.search_method == 'bayesian':
            print(f"贝叶斯优化器已准备好，支持模型: {list(self.bayesian_optimizer.optimizers.keys())}")
    
    def calculate_test_with_bayesian_search(self, X: pd.DataFrame, y: pd.Series, 
                                        model_type: str, n_iterations: int = 20, 
                                        n_folds: int = 5) -> Tuple[float, float, float, float, float, float, float, Dict]:
        """贝叶斯超参数搜索"""
        
        if self.bayesian_optimizer is None:
            print("贝叶斯优化器不可用，退回随机搜索")
            return self.model_optimizer._random_hyperparam_search(X, y, model_type, n_iterations, n_folds)
        
        # 如果当前模型不在优化器中，尝试初始化
        if model_type not in self.bayesian_optimizer.optimizers:
            if hasattr(self.model_optimizer, 'model_configs') and model_type in self.model_optimizer.model_configs:
                param_space = self.model_optimizer.model_configs[model_type]['param_space']
                result = self.bayesian_optimizer.prepare_optimizer(model_type, param_space)
                if not result:
                    print(f"模型 {model_type} 无法初始化贝叶斯优化器，退回随机搜索")
                    return self.model_optimizer._random_hyperparam_search(X, y, model_type, n_iterations, n_folds)
            else:
                print(f"模型 {model_type} 未找到配置，退回随机搜索")
                return self.model_optimizer._random_hyperparam_search(X, y, model_type, n_iterations, n_folds)
        
        # 定义目标函数
        def objective_func(params):
            # 现在calculate_test_score返回7个值
            score, avg_r2, pos_rate, max_r2, perc_error, perc_rmse, avg_mape = self.model_optimizer.calculate_test_score(
                X, y, model_type, params, n_folds
            )
            return score  # 只返回分数用于优化
            
        # 执行贝叶斯优化
        try:
            best_params, _ = self.bayesian_optimizer.optimize(  # 仅接收best_params，丢弃迭代中的best_score
                model_type, 
                objective_func, 
                n_calls=n_iterations,
                verbose=True
            )
            
            # 用最优参数重新计算完整指标（这是唯一的分数来源）
            final_score, avg_r2, pos_rate, max_r2, perc_error, perc_rmse, avg_mape = self.model_optimizer.calculate_test_score(
                X, y, model_type, best_params, n_folds
            )
            
            # 关键修改：打印最终重算的分数，而非迭代中的分数
            print(f"贝叶斯优化完成，最佳分数: {final_score:.4f}")
            
            # 返回所有7个指标 + 最佳参数
            return final_score, avg_r2, pos_rate, max_r2, perc_error, perc_rmse, avg_mape, best_params
            
        except Exception as e:
            print(f"贝叶斯优化失败，退回随机搜索: {e}")
            import traceback
            traceback.print_exc()
            return self.model_optimizer._random_hyperparam_search(X, y, model_type, n_iterations, n_folds)
    def get_all_possible_params(self, model_type: str) -> List[Dict]:
        """获取参数空间中所有可能的组合（兼容性方法）"""
        return self.model_optimizer._get_all_possible_params(model_type)