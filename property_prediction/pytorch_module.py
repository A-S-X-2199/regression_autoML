"""
PyTorch神经网络模块
支持多种网络架构和训练配置
修复了设备不一致的问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')



class PyTorchFNN(nn.Module):
    """PyTorch前馈神经网络"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = (50,), 
                 activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # 创建网络层
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # 添加激活函数
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation.lower() == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            else:
                layers.append(nn.ReLU())
            
            # 添加dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()

        # self._print_device_info()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, x, device=None):
        """预测方法（用于兼容sklearn接口）"""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
            
        with torch.no_grad():
            if isinstance(x, (pd.DataFrame, pd.Series)):
                x = torch.FloatTensor(x.values)
            elif isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            # 确保x是2D
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # 将数据移动到与模型相同的设备
            x = x.to(device)
            prediction = self(x)
            return prediction.cpu().numpy().flatten()



    

class PyTorchDeepFNN(nn.Module):
    """深层PyTorch前馈神经网络"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = (100, 50),
                 activation: str = 'relu', dropout_rate: float = 0.2, 
                 batch_norm: bool = False):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # 创建网络层
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # 批归一化
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # 激活函数
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation.lower() == 'elu':
                layers.append(nn.ELU())
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU(num_parameters=1))
            else:
                layers.append(nn.ReLU())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, x, device=None):
        """预测方法"""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
            
        with torch.no_grad():
            if isinstance(x, (pd.DataFrame, pd.Series)):
                x = torch.FloatTensor(x.values)
            elif isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # 将数据移动到与模型相同的设备
            x = x.to(device)
            prediction = self(x)
            return prediction.cpu().numpy().flatten()


class PyTorchResidualBlock(nn.Module):
    """残差块（输入输出维度必须相同，保持恒等映射）"""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.norm = nn.LayerNorm(input_size)
    
    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.norm(out + identity)
        return self.activation(out)


class PyTorchResNet(nn.Module):
    """残差网络"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = (100, 100, 100),
                 activation: str = 'relu', dropout_rate: float = 0.1):
        super().__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        
        # 残差块
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.residual_blocks.append(
                PyTorchResidualBlock(
                    hidden_sizes[i], 
                    hidden_sizes[i+1],
                    activation,
                    dropout_rate
                )
            )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        
        # 激活函数
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        return self.output_layer(x)
    
    def predict(self, x, device=None):
        """预测方法"""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
            
        with torch.no_grad():
            if isinstance(x, (pd.DataFrame, pd.Series)):
                x = torch.FloatTensor(x.values)
            elif isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # 将数据移动到与模型相同的设备
            x = x.to(device)
            prediction = self(x)
            return prediction.cpu().numpy().flatten()


class PyTorchTrainer:
    """PyTorch模型训练器"""
    
    def __init__(self, model: nn.Module, lr: float = 0.001, 
                 weight_decay: float = 0.0001, device: str = 'cpu', 
                 random_seed: int = 42, validation_fraction: float = 0.2):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.random_seed = random_seed
        self.validation_fraction = validation_fraction


        # 设置随机种子
        self._set_all_random_seeds(random_seed)
        
        # 将模型移到设备
        self.model.to(self.device)
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_history = {'loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def _set_all_random_seeds(self, seed: int):
        """设置所有相关的随机种子"""
        # Python随机
        random.seed(seed)
        
        # NumPy随机
        np.random.seed(seed)
        
        # PyTorch随机
        torch.manual_seed(seed)
        
        # CUDA随机
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def prepare_data(self, X, y, validation_split: bool = True):
        """准备数据"""
        # 转换为tensor
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_tensor = torch.FloatTensor(X.values)
        else:
            X_tensor = torch.FloatTensor(X)
            
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_tensor = torch.FloatTensor(y.values).reshape(-1, 1)
        else:
            y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        if validation_split and self.validation_fraction > 0:
            n_samples = len(X_tensor)
            n_val = int(n_samples * self.validation_fraction)
            
            # 生成随机排列
            g = torch.Generator()
            g.manual_seed(self.random_seed)  # 绑定训练器的随机种子
            indices = torch.randperm(n_samples, generator=g)
            
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            X_val = X_tensor[val_indices]
            y_val = y_tensor[val_indices]
            X_train = X_tensor[train_indices]
            y_train = y_tensor[train_indices]
            
            return X_train, y_train, X_val, y_val
        else:
            return X_tensor, y_tensor, None, None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs: int = 1000, batch_size: int = 32,
            early_stopping: bool = True, patience: int = 50,
            verbose: bool = True):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            epochs: 最大训练轮数
            batch_size: 批大小
            early_stopping: 是否使用早停
            patience: 早停耐心值
            verbose: 是否打印训练信息
        """
        # 准备数据
        if X_val is None or y_val is None:
            X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor = self.prepare_data(
                X_train, y_train, validation_split=early_stopping
            )
        else:
            X_train_tensor, y_train_tensor = self.prepare_data(X_train, y_train, validation_split=False)[:2]
            X_val_tensor, y_val_tensor = self.prepare_data(X_val, y_val, validation_split=False)[:2]
        
        # 创建数据加载器
        g_dataloader = torch.Generator()
        g_dataloader.manual_seed(self.random_seed)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in self.model.modules())
        effective_batch_size = min(batch_size, len(train_dataset))
        if has_bn and effective_batch_size < 2:
            effective_batch_size = min(2, len(train_dataset))
            if effective_batch_size < 2:
                effective_batch_size = 1
                if verbose:
                    print("警告: 训练样本数不足，BatchNorm层将以eval模式运行")
        
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, 
                                 shuffle=True, generator=g_dataloader)
        
        bn_modules = [m for m in self.model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))] if has_bn else []
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                if batch_X.size(0) < 2 and bn_modules:
                    for m in bn_modules:
                        m.eval()
                    outputs = self.model(batch_X)
                    for m in bn_modules:
                        m.train()
                else:
                    outputs = self.model(batch_X)
                
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            
            # 验证阶段
            if early_stopping and X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor_device = X_val_tensor.to(self.device)
                    y_val_tensor_device = y_val_tensor.to(self.device)
                    val_outputs = self.model(X_val_tensor_device)
                    val_loss = self.criterion(val_outputs, y_val_tensor_device).item()
                
                # 保存历史
                self.train_history['loss'].append(avg_train_loss)
                self.train_history['val_loss'].append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch
                    # 保存最佳模型状态
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"早停在轮次 {epoch+1}，最佳轮次 {best_epoch+1}")
                    break
                
                if verbose and (epoch + 1) % 100 == 0:
                    print(f"轮次 {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.6f}, "
                          f"验证损失: {val_loss:.6f}, 耐心: {patience_counter}/{patience}")
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # if verbose:
        #     print(f"训练设备: {self.device}")
        #     print(f"模型所在设备: {next(self.model.parameters()).device}")
        #     if self.device.type == 'cuda':
        #         print(f"GPU内存分配: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        #         print(f"GPU内存缓存: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
        #     print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
        
        return self
    
    def evaluate(self, X, y):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            # 确保X和y在正确的设备上
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X_tensor = torch.FloatTensor(X.values)
            else:
                X_tensor = torch.FloatTensor(X)
                
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_tensor = torch.FloatTensor(y.values).reshape(-1, 1)
            else:
                y_tensor = torch.FloatTensor(y).reshape(-1, 1)
            
            # 移动到正确的设备
            X_tensor = X_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)
            
            predictions = self.model(X_tensor)
            loss = self.criterion(predictions, y_tensor).item()
            
            # 转换为numpy用于计算其他指标
            predictions_np = predictions.cpu().numpy().flatten()
            y_np = y_tensor.cpu().numpy().flatten()
            
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            r2 = r2_score(y_np, predictions_np)
            mse = mean_squared_error(y_np, predictions_np)
            mae = mean_absolute_error(y_np, predictions_np)
            rmse = np.sqrt(mse)
            
            return {
                'loss': loss,
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'predictions': predictions_np
            }
    
    def predict(self, X):
        """预测"""
        # 使用模型的predict方法，并传入设备
        return self.model.predict(X, device=self.device)


class PyTorchConfig:
    """PyTorch配置类"""
    
    @staticmethod
    def get_model_configs(fnn_i: int = 2, fnn_j: int = 3) -> Dict:
        """获取PyTorch模型配置"""
        return {
            'fnn': {
                'i': fnn_i,
                'j': fnn_j,
                'param_space': {
                    'hidden_layer_sizes': [
                        (50,), (100,), (50, 50), (100, 50), 
                        (200,), (100, 100), (50, 25, 10), (100, 50, 25)
                    ],
                    'activation': ['relu','leaky_relu','tanh'],
                    'learning_rate': [0.001, 0.01, 0.005],
                    'weight_decay': [0.0001, 0.001, 0.01],
                    'epochs': [500, 800, 1000],
                    'batch_size_ratio': [0.1, 0.2, 0.3, 0.5],
                    'dropout_rate': [0.0, 0.1, 0.2],
                    'early_stopping': [True],
                    'patience': [50, 100, 150],
                    'validation_fraction': [0.1, 0.15, 0.2],
                    'device': ['cuda']
                }
            },
            'deep_fnn': {
                'i': fnn_i,
                'j': fnn_j,
                'param_space': {
                    'hidden_layer_sizes': [
                        (100, 50), (200, 100), (100, 100, 50), 
                        (200, 100, 50), (100, 50, 25, 10), (200, 100, 50, 25)
                    ],
                    'activation': ['relu', 'leaky_relu', 'prelu'],
                    'learning_rate': [0.001, 0.005],
                    'weight_decay': [0.0001, 0.001],
                    'epochs': [1000, 1500, 2000],
                    'batch_size_ratio': [0.1, 0.2, 0.3],
                    'dropout_rate': [0.1, 0.2, 0.3],
                    'early_stopping': [True],
                    'patience': [100, 150, 200],
                    'validation_fraction': [0.1, 0.15, 0.2],
                    'batch_norm': [True, False],
                    'device': ['cuda']
                }
            },
            'simple_fnn': {
                'i': fnn_i,
                'j': fnn_j,
                'param_space': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'learning_rate': [0.01, 0.05],
                    'weight_decay': [0.001, 0.01],
                    'epochs': [300, 500, 800],
                    'batch_size_ratio': [0.2, 0.3, 0.5],
                    'dropout_rate': [0.0, 0.1],
                    'early_stopping': [False, True],
                    'patience': [50],
                    'validation_fraction': [0.1, 0.15],
                    'device': ['cpu']
                }
            },
            'resnet': {
                'i': fnn_i,
                'j': fnn_j,
                'param_space': {
                    'hidden_layer_sizes': [
                        (100, 100), (200, 200), (100, 100, 100),
                        (200, 200, 200), (100, 100, 100, 100)
                    ],
                    'activation': ['relu'],
                    'learning_rate': [0.001, 0.005],
                    'weight_decay': [0.0001, 0.001],
                    'epochs': [1500, 2000, 2500],
                    'batch_size_ratio': [0.1, 0.2],
                    'dropout_rate': [0.1, 0.2],
                    'early_stopping': [True],
                    'patience': [150, 200],
                    'validation_fraction': [0.1, 0.15],
                }
            }
        }
    
    @staticmethod
    def create_model(model_type: str, input_size: int, params: Dict) -> PyTorchTrainer:
        """创建PyTorch模型训练器"""
        # 提取模型结构参数
        hidden_sizes = params.get('hidden_layer_sizes', (50,))
        activation = params.get('activation', 'relu')
        dropout_rate = params.get('dropout_rate', 0.0)
        batch_norm = params.get('batch_norm', False)
        
        # 创建模型
        if model_type == 'fnn':
            model = PyTorchFNN(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
                dropout_rate=dropout_rate
            )
        elif model_type == 'deep_fnn':
            model = PyTorchDeepFNN(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm
            )
        elif model_type == 'simple_fnn':
            model = PyTorchFNN(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
                dropout_rate=dropout_rate
            )
        elif model_type == 'resnet':
            model = PyTorchResNet(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"不支持的PyTorch模型类型: {model_type}")
        
        # 提取训练参数
        learning_rate = params.get('learning_rate', 0.001)
        weight_decay = params.get('weight_decay', 0.0001)
        device = params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        validation_fraction = params.get('validation_fraction', 0.2)
        
        # 创建训练器
        trainer = PyTorchTrainer(
            model=model,
            lr=learning_rate,
            weight_decay=weight_decay,
            device=device,
            validation_fraction=validation_fraction,
            random_seed=42
        )
        
        # 保存其他参数
        trainer.epochs = params.get('epochs', 1000)
        trainer.batch_size_ratio = params.get('batch_size_ratio', 0.1)
        trainer.early_stopping = params.get('early_stopping', True)
        trainer.patience = params.get('patience', 50)
        
        return trainer


class PyTorchModelSaver:
    """PyTorch模型保存器"""
    
    @staticmethod
    def save_model(model: nn.Module, path: str, config: Dict):
        """保存PyTorch模型"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': config,
            'model_type': config.get('model_type', 'fnn')
        }, path)
    
    @staticmethod
    def load_model(path: str, device: str = 'cpu'):
        """加载PyTorch模型"""
        device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        
        # 根据配置重建模型
        config = checkpoint['model_config']
        model_type = checkpoint.get('model_type', 'fnn')
        input_size = config.get('input_size', 1)
        hidden_sizes = config.get('hidden_sizes', (50,))
        activation = config.get('activation', 'relu')
        dropout_rate = config.get('dropout_rate', 0.0)
        
        # 创建模型
        if model_type == 'fnn':
            model = PyTorchFNN(input_size, hidden_sizes, activation, dropout_rate)
        elif model_type == 'deep_fnn':
            batch_norm = config.get('batch_norm', False)
            model = PyTorchDeepFNN(input_size, hidden_sizes, activation, dropout_rate, batch_norm)
        elif model_type == 'resnet':
            model = PyTorchResNet(input_size, hidden_sizes, activation, dropout_rate)
        else:
            model = PyTorchFNN(input_size, hidden_sizes, activation, dropout_rate)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        
        # 创建训练器包装
        trainer = PyTorchTrainer(
            model=model,
            lr=0.001,
            weight_decay=0.0001,
            device=device,
            random_seed=42
        )
        
        return trainer