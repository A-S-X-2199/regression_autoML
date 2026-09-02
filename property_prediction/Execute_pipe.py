import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import warnings

from . import path_config as pc
# 导入自定义模块
from .data_splitter import split_data_per_property
from .new_data_processor import NewDataProcessor
from .new_data_predictor import MultiPropertyPredictor
from .json_data_processor import JSONDataProcessor
from .merge import merge, merge_train_test_json, merge_train_test_json_update, merge_flat_json_update, img2base64_to_json
from .merge import copy_best_model_jsons, merge_chart_data_json
from .feature_schema import ID_COLUMN, split_feature_columns, display_feature_name
# 忽略无关警告
warnings.filterwarnings('ignore')
import json
import pandas as pd


def _generate_shap_optional(enable_shap: bool = True, **kwargs):
    """SHAP 是附加报告；依赖未安装、分析失败或被配置关闭时均不影响训练结果。"""
    if not enable_shap:
        print("SHAP 可解释性分析已跳过 (train_config.enable_shap=false，如需开启请设为 true)")
        return None
    try:
        from .shap_analysis import generate_shap_all_properties
        return generate_shap_all_properties(**kwargs)
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"SHAP 分析已跳过（可选依赖不可用）: {exc}")
    except Exception as exc:
        print(f"SHAP 分析失败，核心模型结果不受影响: {exc}")
    return None

@dataclass
class PipelineConfig:
    """管道配置类"""
    # 目标配置；为空时由输入数据或已有模型确定。
    properties: Optional[List[str]] = None
    property_indices: Optional[List[int]] = None

    # 训练配置
    model_list_custom: Optional[List[str]] = None
    model_type: Union[int, str] = 'plus'  # 0:低复杂度, 1:低+中复杂度, 2:高复杂度, 'plus':推荐(精简集), 'custom':自定义, 其他:全部
    test_size: float = 0.2  # 0则自动切换为全量训练
    n_iterations: int = 20
    n_folds: int = 4
    random_seed: int = 100  # 随机种子
    judge: int = 0
    max_no_improve_rounds: int = 0  # judge=2 递归剔除时，test因子未提升后允许继续剔除的轮数上限
    search_method: str = 'random'  # 'random' or 'bayesian'
    nn_backend: str = 'sklearn'  # 'sklearn' or 'pytorch'
    n_jobs: int = -1  # 核数控制：-1使用所有核,>0指定核数

    # 特征工程配置
    reduction_type: str = 'custom'  # 'none', 'pca', 'pls', 'tsvd', 'custom'
    mode_list: List[str] = None

    # 调试配置
    debug_dump_csv: bool = False  # 标准化前将特征+目标拼接输出CSV及t-SNE可视化
    outlier_n: int = None  # 异常点标记数,不为None且>0时在散点图标红测试集最大误差的n个点

    # 异常样本剔除配置
    remove_outliers: bool = False  # 是否启用学生化残差异常点剔除
    outlier_std_threshold: float = 2.0  # 学生化残差阈值,默认2倍标准差
    outlier_model: str = 'ridge'  # 代理模型类型, 'ridge' 或 'huber'

    # 图片输出配置
    enable_figures_base64: bool = False  # 是否生成base64图片（figures.json并合并到output.json），默认关闭；前端直接用chart_data.json数据文件生成图表

    # SHAP 可解释性配置
    enable_shap: bool = True  # 是否在训练评估后生成 SHAP 分析（大样本下 KernelExplainer 耗时明显，可设 false 跳过）

    def __post_init__(self):
        """初始化默认值+参数校验"""
        # 降维模式列表默认值
        if self.mode_list is None:
            self.mode_list = ['pls', 'tsvd', 'pca']
        # 校验test_size范围：0<=test_size<1
        if not 0 <= self.test_size < 1:
            raise ValueError(f"test_size必须在[0,1)范围内,当前值：{self.test_size}")
        # 校验核数
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError(f"n_jobs必须为-1或正整数,当前值：{self.n_jobs}")
        # 校验搜索方法
        if self.search_method not in ['random', 'bayesian']:
            raise ValueError(f"search_method仅支持'random'/'bayesian',当前值：{self.search_method}")
        # 校验神经网络后端
        if self.nn_backend not in ['sklearn', 'pytorch']:
            raise ValueError(f"nn_backend仅支持'sklearn'/'pytorch',当前值：{self.nn_backend}")


class ModelSelector:
    """模型选择器类"""
    @staticmethod
    def get_model_list(model_type: Union[int, str], model_list_custom: List[str] = None) -> List[str]:
        """根据模型类型获取模型列表
        model_type: 0/1/2 为数字档位；'plus'/'custom' 与 reduction_type 命名风格统一
        """
        # 定义所有模型类别
        model_list_low = [
            'linear', 'ridge', 'lasso', 'elasticnet', 'bayesian_ridge',
            'dt', 'linearsvr', 'knn', 'huber', 'poly'
        ]

        model_list_medium = [
            'svr', 'svr_rbf', 'rf', 'extra_trees', 'gbr',
            'gbdt', 'hist_gbdt', 'adaboost', 'gpr',
            'xgb', 'lgbm', 'catboost'
        ]

        model_list_high = ['fnn', 'deep_fnn', 'resnet']

        # 通用回归推荐模型集，覆盖线性、核方法、树集成、高斯过程和神经网络。
        model_list_recommended = [
            'ridge',       # 正则化线性回归：稳定且能处理特征共线性
            'lasso',       # L1稀疏线性回归：自动特征筛选
            'huber',       # Huber回归：抗离群点线性基线
            'svr_rbf',     # RBF核SVM：小样本非线性拟合
            'rf',          # 随机森林：bagging树集成，抗噪
            'extra_trees', # 极端随机树：比RF更随机的bagging变体，方差更低
            'xgb',         # XGBoost：提升树
            'lgbm',        # LightGBM：轻量提升树
            'gpr',         # 高斯过程：小样本+不确定性估计
            'fnn',         # 神经网络：高容量拟合
        ]

        model_list_custom = model_list_custom if model_list_custom is not None else [
            'svr', 'lgbm', 'xgb', 'ridge', 'fnn'
        ]

        # 根据类型选择
        if model_type in (0, 'low'):
            return model_list_low
        elif model_type in (1, 'medium'):
            return model_list_low + model_list_medium
        elif model_type in (2, 'high'):
            return model_list_high
        elif model_type in ('plus', 3):  # 3为兼容旧配置
            return model_list_recommended
        elif model_type in ('custom', 4):  # 4为兼容旧配置
            return model_list_custom
        else:
            # 返回所有模型
            return model_list_low + model_list_medium + model_list_high


class GenericMLPipeline:
    """通用多目标回归管道主类。"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.converter = JSONDataProcessor()
        self.model_selector = ModelSelector()
        # 初始化时从数据库读取性能列表
        self._init_properties()
        # 核数配置全局生效(设置环境变量,适配sklearn/xgboost/lgbm等)

    def set_global_n_jobs(self):
        """设置全局并行核数,适配所有主流计算库,必须在导入sklearn/xgb/lgbm前执行"""
        n_jobs = self.config.n_jobs
        n_threads = str(n_jobs) if n_jobs > 0 else "1"  
        # 1. 通用线性代数库(覆盖OMP/MKL/OPENBLAS等所有底层库)
        os.environ['OMP_NUM_THREADS'] = n_threads
        os.environ['MKL_NUM_THREADS'] = n_threads
        os.environ['NUMEXPR_NUM_THREADS'] = n_threads
        os.environ['OPENBLAS_NUM_THREADS'] = n_threads
        os.environ['BLIS_NUM_THREADS'] = n_threads
        os.environ['VECLIB_MAXIMUM_THREADS'] = n_threads
        # 2. 机器学习库专属(XGBoost/LightGBM/CatBoost)
        os.environ['XGBOOST_NUM_THREADS'] = n_threads
        os.environ['LIGHTGBM_NUM_THREADS'] = n_threads
        os.environ['CATBOOST_NUM_THREADS'] = n_threads
        print(f"全局并行核数已设置为: {n_threads}(底层库+ML库统一约束)")


    def _init_properties(self):
        """从数据库初始化性能列表"""
        print("=" * 30, "初始化性能列表", "=" * 30)
        # 从路径配置模块读取训练JSON路径
        db_properties = self.converter.get_properties_list(pc.TRAIN_JSON)

        if self.config.properties is None:
            # 使用数据库中的性能列表
            self.config.properties = db_properties
            print(f"从数据库加载性能指标数量: {len(db_properties)} 个")

        # 设置property_indices(默认使用所有)
        if self.config.property_indices is None:
            self.config.property_indices = list(range(len(self.config.properties)))

        # 显示选中的性能指标
        selected_props = [self.config.properties[i] for i in self.config.property_indices]
        print(f"最终选中训练/评估/预测的性能指标 ({len(selected_props)} 个): {', '.join(selected_props)}")
        print("=" * 70)

    def _prepare_data(self):
        """准备通用训练数据，返回完整数据供各目标独立划分。"""
        print("=" * 30, "开始数据预处理", "=" * 30)
        test_size = self.config.test_size
        if test_size == 0:
            print(f"test_size=0，采用【全量训练模式】，所有数据用于训练，无测试集")
        else:
            print(f"采用【逐性质独立划分模式】，测试集比例: {test_size}, 随机种子: {self.config.random_seed}")
            print(f"注意：每个性质将根据自身NaN分布独立划分训练/测试集")

        print("正在处理JSON训练数据...")
        Features_df, Targets_df = self.converter.process_training_data(
            self.config.properties, pc.TRAIN_JSON, pc.CORR_BAR_PLOT
        )

        # ========== 异常样本剔除（基于 Ridge + 学生化残差，分项点独立处理） ==========
        if self.config.remove_outliers:
            outlier_vis_dir = pc.VISUAL_ROOT / "outlier_diagnostics"
            from .studentized_residual_filter import filter_outliers_per_property
            Features_df, Targets_df, self.outlier_info = filter_outliers_per_property(
                features_df=Features_df,
                targets_df=Targets_df,
                property_names=self.config.properties,
                std_threshold=self.config.outlier_std_threshold,
                outlier_model=self.config.outlier_model,
                random_state=self.config.random_seed,
                output_dir=outlier_vis_dir,
            )
        else:
            self.outlier_info = {}

        # ========== 调试：标准化前拼接特征+目标输出CSV + t-SNE可视化 + 相关性热力图 ==========
        if self.config.debug_dump_csv:
            dump_path = Path(pc.OUTPUT_ROOT) / "debug_features_targets_pre_scale.csv"
            Path(pc.OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
            df_merged = Features_df.merge(Targets_df, on=ID_COLUMN, how='inner')
            df_merged.to_csv(str(dump_path), index=False, encoding='utf-8-sig')
            print(f"[DEBUG] 标准化前特征+目标已导出: {dump_path}")
            print(f"        形状: {df_merged.shape[0]} 行 × {df_merged.shape[1]} 列")

            self._debug_tsne_plot(Features_df, Targets_df, self.config.properties)
            self._debug_corr_heatmap(Features_df, Targets_df, self.config.properties)

        # 计算特征统计信息（供后续使用，但划分在逐性质循环中进行）
        feature_columns = Features_df.columns[1:]  # 排除ID列
        front_columns, back_columns = split_feature_columns(Features_df.columns)
        num_front_features = len(front_columns)
        num_back_features = len(back_columns)

        # 保存完整数据到实例（供其他方法如SHAP分析、评估使用）
        self.full_features_df = Features_df
        self.full_targets_df = Targets_df

        print(f"数据预处理完成，总样本数: {len(Features_df)}")
        print(f"特征统计: 前部分={num_front_features}, 后部分(参与降维)={num_back_features}")
        print("=" * 70)
        return num_front_features, num_back_features, Features_df, Targets_df

    def _debug_tsne_plot(self, Features_df, Targets_df, property_names):
        """
        调试：对标准化前的特征做 t-SNE 降维,按每个性质值着色绘图,
        并在图中显示量化指标：Spearman ρ、局部邻域方差比、Moran's I、离散轮廓系数。
        """
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import spearmanr
        from scipy.spatial.distance import cdist
        from sklearn.metrics import silhouette_score

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        df_merged = Features_df.merge(Targets_df, on=ID_COLUMN, how='inner')
        feature_cols = [c for c in Features_df.columns if c != ID_COLUMN]
        X = df_merged[feature_cols].values.astype(np.float64)

        if X.shape[0] < 5:
            print("[DEBUG] t-SNE: 样本数不足5个,跳过")
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        perplexity = min(30, X.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)

        vis_dir = Path(pc.VISUAL_ROOT) / "debug_tsne"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for prop in property_names:
            if prop not in df_merged.columns:
                continue

            values = df_merged[prop].values.astype(np.float64)
            mask = ~np.isnan(values)
            n_valid = mask.sum()
            if n_valid < 5:
                continue

            tsne_coords = X_tsne[mask]
            vals = values[mask]

            # 1. Spearman ρ
            rho_x, _ = spearmanr(tsne_coords[:, 0], vals)
            rho_y, _ = spearmanr(tsne_coords[:, 1], vals)
            spearman_max = max(abs(rho_x), abs(rho_y))

            # 2. 局部邻域方差比 (k=3)
            k = min(3, n_valid - 1)
            dists = cdist(tsne_coords, tsne_coords)
            nn_indices = np.argpartition(dists, k + 1, axis=1)[:, 1:k + 1]
            local_var = np.mean([np.var(vals[nn_indices[i]]) for i in range(n_valid)])
            global_var = np.var(vals)
            nn_var_ratio = local_var / global_var if global_var > 1e-10 else 1.0

            # 3. Moran's I
            eps = 1e-10
            w = 1.0 / (dists + eps)
            np.fill_diagonal(w, 0)
            w_sum = w.sum()
            v_centered = vals - vals.mean()
            moran_num = n_valid * np.sum(w * np.outer(v_centered, v_centered))
            moran_den = w_sum * np.sum(v_centered ** 2)
            moran_i = moran_num / moran_den if moran_den > 1e-10 else 0.0

            # 4. 离散化轮廓系数
            n_bins = min(3, n_valid - 1)
            if n_bins >= 2 and np.ptp(vals) > 1e-10:
                labels = np.digitize(vals, np.linspace(vals.min(), vals.max(), n_bins + 1)[1:-1])
                n_unique = len(np.unique(labels))
                sil = silhouette_score(tsne_coords, labels) if n_unique >= 2 and n_unique < n_valid else 0.0
            else:
                sil = 0.0

            # ---- 绘图 ----
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                tsne_coords[:, 0], tsne_coords[:, 1],
                c=vals, cmap='viridis', s=60, alpha=0.85, edgecolors='k', linewidths=0.3
            )
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.78)
            cbar.set_label(prop, fontsize=12)

            ax.set_title(f't-SNE: {prop}', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE-1')
            ax.set_ylabel('t-SNE-2')

            ids = df_merged[ID_COLUMN].values[mask]
            for i, rid in enumerate(ids):
                ax.annotate(str(rid), (tsne_coords[i, 0], tsne_coords[i, 1]),
                            fontsize=5, alpha=0.6, ha='center', va='bottom')

            # 量化指标 + 判定(左下角)
            metrics_text = (
                f"Spearman $\\rho$ (max|x,y|): {spearman_max:.4f}\n"
                f"邻域方差比 (k={k}): {nn_var_ratio:.4f}\n"
                f"Moran's I: {moran_i:.4f}\n"
                f"离散轮廓系数: {sil:.4f}"
            )
            if spearman_max > 0.4 and nn_var_ratio < 0.4:
                hint = "推测: 结构较明显,预测前景良好"
                color = 'green'
            elif nn_var_ratio > 0.8 or spearman_max < 0.2:
                hint = "推测: 分布接近随机,预测难度较大"
                color = 'red'
            else:
                hint = "推测: 结构中等,预测待验证"
                color = 'orange'

            ax.text(
                0.02, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray')
            )
            ax.text(0.5, 0.02, hint, transform=ax.transAxes,
                    fontsize=9, color=color, fontstyle='italic')

            plt.tight_layout()
            safe_name = prop.replace('/', '_').replace('\\', '_').replace(':', '_')
            save_path = vis_dir / f"tsne_{safe_name}.png"
            plt.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"[DEBUG] t-SNE 图已保存: {save_path}  "
                  f"(Spearman={spearman_max:.3f}, 方差比={nn_var_ratio:.3f}, "
                  f"MoranI={moran_i:.3f}, Sil={sil:.3f})")

        print(f"[DEBUG] t-SNE 可视化完成,共 {len(property_names)} 张图,保存在 {vis_dir}")

    def _debug_corr_heatmap(self, Features_df, Targets_df, property_names):
        """
        调试：绘制目标与所有特征的皮尔逊相关性热力图。
        行=目标，列=特征（显示时去掉 FRONT::/BACK:: 分组前缀）。
        """
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        df_merged = Features_df.merge(Targets_df, on=ID_COLUMN, how='inner')
        feature_cols = [c for c in Features_df.columns if c != ID_COLUMN]
        target_cols = [p for p in property_names if p in df_merged.columns]

        if not feature_cols or not target_cols:
            print("[DEBUG] 相关性热力图: 特征或目标列为空,跳过")
            return

        def _short_label(col):
            return display_feature_name(col)

        feat_labels = [_short_label(c) for c in feature_cols]

        n_feat = len(feature_cols)
        n_targ = len(target_cols)
        corr_matrix = np.zeros((n_targ, n_feat))
        for i, tcol in enumerate(target_cols):
            for j, fcol in enumerate(feature_cols):
                valid = df_merged[[fcol, tcol]].dropna()
                if len(valid) >= 3:
                    corr_matrix[i, j] = valid[fcol].corr(valid[tcol])

        fig_w = max(16, n_feat * 0.35)
        fig_h = max(5, n_targ * 0.9)
        fontsize_cell = max(13, min(11, 200 / n_feat))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)

        ax.set_xticks(range(n_feat))
        ax.set_xticklabels(feat_labels, rotation=90, fontsize=fontsize_cell, ha='center')
        ax.set_yticks(range(n_targ))
        ax.set_yticklabels(target_cols, fontsize=fontsize_cell)

        ax.set_title('特征-目标 皮尔逊相关系数热力图', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Pearson r', fontsize=11)

        plt.tight_layout()
        vis_dir = Path(pc.VISUAL_ROOT) / "debug_tsne"
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_path = vis_dir / "corr_heatmap.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[DEBUG] 相关性热力图已保存: {save_path}")

    def _check_test_file_valid(self):
        """检查是否有有效的逐性质测试集"""
        if self.config.test_size == 0:
            return False
        # 从 optimizer 或 pipeline 自身获取逐性质测试数据
        for source in [getattr(self, '_per_property_test_data', {}),
                       getattr(getattr(self, '_optimizer', None), '_per_property_test_data', {})]:
            if source:
                for test_data in source.values():
                    if len(test_data.get('test_ids', [])) > 0:
                        return True
        return False

    def train_full_model(self):
        """训练完整模型（逐性质独立划分模式）"""
        print("=" * 30, "开始全量模型训练", "=" * 30)

        # 项点更新：不再清空整个 output 目录，已有模型/评估记录（含其他项点）全部保留。
        # 仅对本次选中的项点做定向清理（由 optimizer.main 的 clean_model_directory 完成）。
        pc.init_paths()

        # 训练前删除选中项点的旧模型信息，防止某些项点训练失败后旧模型信息遗漏/误用
        self._clean_selected_model_files(
            [self.config.properties[i] for i in self.config.property_indices])

        orig_features, reduction_features, Features_df, Targets_df = self._prepare_data()
        model_list = self.model_selector.get_model_list(
            self.config.model_type, self.config.model_list_custom
        )
        print(f"本次训练使用模型列表 ({len(model_list)} 个): {', '.join(model_list)}")
        print(f"超参数优化迭代次数: {self.config.n_iterations}, 交叉验证折数: {self.config.n_folds}")
        print(f"超参数搜索方法: {self.config.search_method}, 神经网络后端: {self.config.nn_backend}")

        # 训练器依赖 xgboost 等可选训练库；仅在全量训练时加载，
        # 避免纯预测流程因未安装训练依赖而无法启动。
        from .model_optimizer_main import ModelOptimizer
        optimizer = ModelOptimizer()
        # 任一开关为 True 即开启 base64 输出（optimizer 侧配置优先于 config，不做覆盖）
        optimizer.enable_figures_base64 = bool(
            optimizer.enable_figures_base64 or self.config.enable_figures_base64
        )
        all_results, test_performance_df, best_models_info = optimizer.main(
            output_dir=pc.OUTPUT_ROOT,
            training_params_dir=pc.TRAIN_PARAMS_ROOT,
            property_names=self.config.properties,
            judge=self.config.judge,
            max_no_improve_rounds=self.config.max_no_improve_rounds,
            property_indices=self.config.property_indices,
            model_list=model_list,
            best_model_dir=pc.BEST_MODEL_ROOT,
            n_iterations=self.config.n_iterations,
            n_folds=self.config.n_folds,
            reduction_type=self.config.reduction_type,
            reduction_features=reduction_features,
            orig_features=orig_features,
            mode_list=self.config.mode_list,
            search_method=self.config.search_method,
            nn_backend=self.config.nn_backend,
            train_features_df=Features_df,
            train_targets_df=Targets_df,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
        )
        # 保存optimizer引用，同时复制逐性质测试数据到pipeline
        self._optimizer = optimizer
        self._per_property_test_data = dict(optimizer._per_property_test_data)
        print(f"全量模型训练完成，成功处理性能指标数量: {len(all_results)} 个")
        # 持久化本次训练的各项点测试集划分，供后续项点更新时未重训项点复用
        self._save_property_test_splits()
        
        return all_results, test_performance_df, best_models_info

    def train_incremental_model(self):
        """增量训练模型（逐性质独立划分模式）"""
        print("=" * 30, "开始增量模型训练", "=" * 30)
        orig_features, reduction_features, Features_df, Targets_df = self._prepare_data()
        target_properties = [self.config.properties[i] for i in self.config.property_indices]
        print(f"增量训练目标性能指标 ({len(target_properties)} 个): {', '.join(target_properties)}")
        # 注意：增量训练不得删除选中项点的已有模型文件——它依赖已有 best_model_info
        # 中的模型与降维配置引导训练（load_existing_model_info），
        # 且模型文件会在训练成功后由 overwrite_model_files 覆盖更新。
        # 删除仅在全量训练（train_full_model）中执行，防止训练失败后旧模型信息遗漏。

        # 增量训练同样按需加载其训练依赖。
        from .incremental_trainer import incremental_train_main
        results = incremental_train_main(
            orig_features=orig_features,
            reduction_features=reduction_features,
            model_dir=pc.BEST_MODEL_ROOT,
            training_params_dir=pc.TRAIN_PARAMS_ROOT,
            property_names=target_properties,
            output_dir=pc.INCREMENTAL_OUTPUT_ROOT,
            n_iterations=self.config.n_iterations,
            n_folds=self.config.n_folds,
            search_method=self.config.search_method,
            batch_mode=True,
            judge=self.config.judge,
            max_no_improve_rounds=self.config.max_no_improve_rounds,
            backup_original=False,
            train_features_df=Features_df,
            train_targets_df=Targets_df,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
        )
        # 将增量训练的逐性质测试数据存储到 pipeline 实例，供 evaluate_model 使用
        train_results, per_property_test_data = results
        if not hasattr(self, '_per_property_test_data') or self._per_property_test_data is None:
            self._per_property_test_data = {}
        self._per_property_test_data.update(per_property_test_data)
        print(f"增量模型训练完成，训练结果数量: {len(train_results)} 个")
        # 持久化本次训练的各项点测试集划分，供后续项点更新时未重训项点复用
        self._save_property_test_splits()
        return train_results

    def _clean_selected_model_files(self, target_properties: List[str]):
        """训练前删除选中项点的旧模型/参数/测试集划分文件（定向清理）。

        覆盖三个目录：
          - artifacts/property_prediction/best_models
          - artifacts/property_prediction/reduction_params
          - artifacts/property_prediction/final_models_params
        防止某些项点训练失败后仍残留旧模型信息而被误用/遗漏；
        其他项点文件及 best_model_perform.json 等全局结果不受影响。
        训练成功后会重新生成对应文件。
        """
        best_dir = pc.BEST_MODEL_ROOT
        params_dir = pc.TRAIN_PARAMS_ROOT
        final_dir = pc.FINAL_MODEL_ROOT
        deleted = []
        for prop in target_properties:
            safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")
            prefixes = [
                f'best_model_{safe_prop}',
                f'best_model_info_{safe_prop}',
                f'best_features_{safe_prop}',
                f'best_params_{safe_prop}',
                f'test_split_{safe_prop}',
            ]
            for d in (best_dir, params_dir, final_dir):
                if not os.path.isdir(str(d)):
                    continue
                for f in os.listdir(str(d)):
                    hit = any(f.startswith(p) for p in prefixes)
                    if not hit and d in (params_dir, final_dir):
                        # 降维/标准化参数文件命名形如 {type}_params_{safe_prop}.pkl
                        # （final_models_params 中亦含 standardization_params_{safe_prop}.pkl）
                        hit = (f.endswith(f'params_{safe_prop}.pkl')
                               or f.endswith(f'_{safe_prop}.pkl'))
                    if hit:
                        path = os.path.join(str(d), f)
                        try:
                            os.remove(path)
                            deleted.append(f)
                        except OSError as e:
                            print(f"  警告: 删除 {f} 失败: {e}")
        if deleted:
            print(f"已删除选中项点的旧模型/参数文件 {len(deleted)} 个: {deleted}")

    def _save_property_test_splits(self):
        """持久化本次训练/评估使用的逐项点测试集划分到 best_models 目录（test_split_{prop}.json）。

        项点更新场景下，未重训项点将复用该历史划分进行评估，
        从而不受本次修改的分割种子/测试比例影响（保证“其他项点正常保留”）。
        """
        test_data_source = getattr(self, '_per_property_test_data', {})
        if not test_data_source:
            return
        os.makedirs(str(pc.BEST_MODEL_ROOT), exist_ok=True)
        for prop, td in test_data_source.items():
            test_ids = td.get('test_ids', [])
            y_test = td.get('y_test')
            if not test_ids:
                continue
            y_values = []
            if y_test is not None:
                try:
                    y_values = [None if v is None or v != v else float(v) for v in y_test.tolist()]
                except Exception:
                    y_values = []
            safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")
            split_file = Path(pc.BEST_MODEL_ROOT) / f"test_split_{safe_prop}.json"
            try:
                with open(split_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'property': prop,
                        'test_ids': [str(t) for t in test_ids],
                        'y_test': y_values,
                        'random_seed': self.config.random_seed,
                        'test_size': self.config.test_size,
                    }, f, ensure_ascii=False, indent=2)
                print(f"  已保存项点 '{prop}' 的测试集划分: {split_file} ({len(test_ids)} 个样本)")
            except Exception as e:
                print(f"  保存项点 '{prop}' 测试集划分失败: {e}")

    def _load_property_test_split(self, prop: str) -> Optional[Dict]:
        """加载项点历史保存的测试集划分；不存在或无效时返回 None。"""
        safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")
        split_file = Path(pc.BEST_MODEL_ROOT) / f"test_split_{safe_prop}.json"
        if not split_file.exists():
            return None
        try:
            with open(split_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            test_ids = data.get('test_ids', [])
            y_values = data.get('y_test', [])
            if not test_ids:
                return None
            y_test = pd.Series([float('nan') if v is None else v for v in y_values])
            return {'test_ids': test_ids, 'X_test_scaled': None, 'y_test': y_test}
        except Exception as e:
            print(f"  加载项点 '{prop}' 历史测试集划分失败: {e}")
            return None

    def evaluate_model(self):
        """评估模型并生成可视化（基于逐性质独立测试集）

        项点更新场景下执行“统一完整评估”：
        - 评估范围 = 本次选中的项点 ∪ 已有性能预测模型的全部项点
        - 本次参与训练的项点使用其本次划分的测试集；
        - 其余项点（未重训）优先加载历史保存的测试集划分（不受本次分割种子/比例影响），
          仅在无历史数据时按当前配置重新划分（打印警告）。
        """
        target_properties = [self.config.properties[i] for i in self.config.property_indices]

        # 完整评估：本次选中的项点 + 已有性能预测模型的全部项点
        all_model_props = set()
        if os.path.isdir(str(pc.BEST_MODEL_ROOT)):
            for f in os.listdir(str(pc.BEST_MODEL_ROOT)):
                if f.startswith('best_model_info_') and f.endswith('.json'):
                    all_model_props.add(f.replace('best_model_info_', '').replace('.json', ''))
        eval_properties = sorted(set(target_properties) | all_model_props)

        # 最终模型训练范围：仅限配置中指定的项点（properties 对应的所选索引），
        # 避免数据集更新后用新数据污染未指定的历史模型；
        # 配置为空（未指定任何项点）时退化为全部有模型的项点。
        final_train_props = list(target_properties) if target_properties else eval_properties

        if not self._check_test_file_valid():
            print("=" * 30, "模型评估跳过", "=" * 30)
            print("原因：未检测到有效测试集(test_size=0或数据为空)")
            # 评估跳过仍基于全量数据训练最终模型（test_size=0 时全量即训练集），供预测调用
            self._train_final_models(final_train_props)
            print("=" * 70)
            return False
        
        print("=" * 30, "开始模型评估与可视化（逐性质独立测试集）", "=" * 30)
        print(f"完整评估目标性能指标 ({len(eval_properties)} 个): {', '.join(eval_properties)}")

        # 获取逐性质测试数据（本次训练划分的）
        if not hasattr(self, '_per_property_test_data') or self._per_property_test_data is None:
            self._per_property_test_data = dict(
                getattr(getattr(self, '_optimizer', None), '_per_property_test_data', {}))
        test_data_source = self._per_property_test_data

        # 未参与本次训练的项点：优先复用历史保存的测试集划分（不受本次分割种子/比例影响）
        for prop in eval_properties:
            if prop in test_data_source:
                continue
            loaded = self._load_property_test_split(prop)
            if loaded is not None:
                test_data_source[prop] = loaded
                print(f"  项点 '{prop}' 未参与本次训练，已复用历史保存的测试集划分")
                continue
            if self.full_features_df is None or self.full_targets_df is None:
                print(f"  警告: 项点 '{prop}' 无完整数据，无法补充测试集，跳过评估")
                continue
            try:
                print(f"  警告: 项点 '{prop}' 无历史测试集划分，将按当前配置重新划分"
                      f"（若修改了分割种子/测试比例，其评估结果将与历史不同）")
                X_train_scaled, y_train, X_test_scaled, y_test = split_data_per_property(
                    df_features=self.full_features_df,
                    df_targets=self.full_targets_df,
                    property_name=prop,
                    standardization_params_dir=str(pc.TRAIN_PARAMS_ROOT),
                    test_size=self.config.test_size,
                    random_state=self.config.random_seed,
                    standardize=True,
                )
                id_col = self.full_features_df.columns[0]
                test_data_source[prop] = {
                    'test_ids': X_test_scaled[id_col].tolist() if len(X_test_scaled) > 0 else [],
                    'X_test_scaled': X_test_scaled,
                    'y_test': y_test,
                }
                print(f"  项点 '{prop}' 已按当前配置补充测试集")
            except Exception as e:
                print(f"  警告: 项点 '{prop}' 补充测试集失败: {e}")

        # 将 per-property test data 中的实际值提取出来，构建 test_actual_df
        id_col = self.full_features_df.columns[0]
        all_test_ids = set()
        prop_actual_series = {}
        for prop in eval_properties:
            test_data = test_data_source.get(prop, {})
            test_ids = test_data.get('test_ids', [])
            y_test = test_data.get('y_test')
            if y_test is not None and len(y_test) > 0 and len(test_ids) > 0:
                prop_actual_series[prop] = pd.Series(y_test.values, index=test_ids, name=prop)
                all_test_ids.update(test_ids)
            else:
                prop_actual_series[prop] = pd.Series(dtype=float, name=prop)

        # 构建完整的 test_actual_df（兼容可视化需求）
        test_actual_records = []
        for tid in sorted(all_test_ids):
            row = {id_col: tid}
            for prop, s in prop_actual_series.items():
                row[prop] = s.get(tid, float('nan'))
            test_actual_records.append(row)
        test_actual_df = pd.DataFrame(test_actual_records) if test_actual_records else pd.DataFrame()

        # 使用 NewDataProcessor 处理完整特征数据（逐性质标准化 + 降维）
        processor = NewDataProcessor(
            new_data_df=self.full_features_df,
            best_models_dir=pc.BEST_MODEL_ROOT,
            output_path=str(pc.PROCESSED_TRAIN_JSON),
            training_params_dir=pc.TRAIN_PARAMS_ROOT,
            property_names=eval_properties,
        )
        processor.run_pipeline()
        test_features_dict = dict(processor.final_results_dict)

        # 使用 MultiPropertyPredictor 预测
        predictor = MultiPropertyPredictor(
            test_features_df=test_features_dict,
            best_models_dir=pc.BEST_MODEL_ROOT,
            output_path=str(pc.TEST_PREDICT_RESULT_JSON),
            apply_non_negative=False,
            apply_range_constraint=True,
            json_output_path=str(pc.JSON_RESULT_PREDICT),
            train_data_df=self.full_targets_df,
        )
        predictor.set_property_list(eval_properties)
        predictor.run_pipeline()
        print(f"测试集预测完成,结果保存至: {pc.TEST_PREDICT_RESULT_JSON.absolute()}")
        pred_df = predictor.prediction_result_df

        # 过滤预测结果：仅保留测试集行
        id_match_col = None
        for col in ['样本编号', '样本标识', '原始编号']:
            if col in pred_df.columns:
                id_match_col = col
                break
        if id_match_col and len(all_test_ids) > 0:
            pred_df_test = pred_df[pred_df[id_match_col].astype(str).isin(
                [str(tid) for tid in all_test_ids]
            )].reset_index(drop=True)
            print(f"过滤预测结果为仅测试集: {pred_df_test.shape}")
        else:
            pred_df_test = pred_df

        # 按性质测试集划分（split_data_per_property 的独立划分），供可视化/图表数据按性质呈现
        per_prop_test_ids = {
            prop: [str(x) for x in (test_data_source.get(prop, {}).get('test_ids') or [])]
            for prop in eval_properties
            if test_data_source.get(prop, {}).get('test_ids')
        }

        self._generate_visualizations(
            eval_properties,
            pred_df_test,
            test_actual_df,
            test_ids_per_prop=per_prop_test_ids,
        )
        print("模型评估与可视化完成！")
        print("=" * 70)
        merge(str(pc.JSON_RESULT_TRAIN_ROOT), "output_test.json")

        # 将最终测试集 R²/MAPE 写回 best_model_info，供预测输出 model_info 展示模型泛化能力
        self._backfill_test_metrics_to_best_models()

        # 评估完成后：仅对配置中指定的项点，基于全量数据（train+test 全部样本）重新训练最终模型，
        # 标准化/降维/异常点剔除参数保存至性能预测 artifacts 供预测调用
        # （未指定的历史项点不做最终训练，避免数据集更新后对新数据产生污染）
        self._train_final_models(final_train_props)
        return True

    def _train_final_models(self, properties: List[str]):
        """基于 best_models 中的模型信息，用全量数据重新训练最终模型。

        最终模型和预处理参数保存到 artifacts/property_prediction，预测时读取。
        """
        print("=" * 30, "开始全量数据最终模型训练", "=" * 30)
        if getattr(self, 'full_features_df', None) is None or getattr(self, 'full_targets_df', None) is None:
            print("警告: 无完整特征/目标数据，跳过最终模型训练")
            return
        try:
            from .final_model_trainer import train_final_models
            train_final_models(
                features_df=self.full_features_df,
                targets_df=self.full_targets_df,
                best_models_dir=str(pc.BEST_MODEL_ROOT),
                final_models_dir=str(pc.FINAL_MODEL_ROOT),
                properties=properties,
            )
        except Exception as e:
            import traceback
            print(f"最终模型训练失败: {e}")
            print(traceback.format_exc())
        print("=" * 30, "全量数据最终模型训练完成", "=" * 30)

    def predict_new_data(self):
        """独立预测模式：对新数据进行预测并保存结果"""
        print("=" * 30, "启动独立预测模式", "=" * 30)
        target_properties = [self.config.properties[i] for i in self.config.property_indices]
        print(f"预测目标性能指标 ({len(target_properties)} 个): {', '.join(target_properties)}")

        print("正在处理JSON预测数据...")
        predict_features_df = self.converter.process_prediction_data(pc.PREDICT_JSON)

        train_data = self.converter.load_json_data(pc.TRAIN_JSON)
        if train_data is None or 'generic_targets' not in train_data:
            print("错误: 无法加载训练数据或数据格式不正确,预测终止")
            return False
        from .feature_schema import build_target_frame
        train_targets_df = build_target_frame(
            train_data['generic_targets'],
            list(train_data.get('generic_features', {}).keys()),
        )

        processor = NewDataProcessor(
            predict=True,
            new_data_df=predict_features_df,
            best_models_dir=pc.BEST_MODEL_ROOT,
            output_path=str(pc.PROCESSED_PREDICT_JSON),
            training_params_dir=pc.TRAIN_PARAMS_ROOT,
            standardization_params_path=pc.STANDARDIZATION_PARAMS,
            property_names=target_properties,
            final_models_dir=pc.FINAL_MODEL_ROOT,
        )
        processor.run_pipeline()
        print(f"新数据预处理完成,处理后数据保存至: {pc.PROCESSED_PREDICT_JSON.absolute()}")

        predict_features_dict = dict(processor.final_results_dict)

        predictor = MultiPropertyPredictor(
            predict=True,
            test_features_df=predict_features_dict,
            best_models_dir=pc.BEST_MODEL_ROOT,
            output_path=str(pc.NEW_DATA_PREDICT_RESULT_JSON),
            apply_non_negative=False,
            apply_range_constraint=False,
            json_output_path=str(pc.JSON_RESULT_PREDICT),
            train_data_df=train_targets_df,
            final_models_dir=pc.FINAL_MODEL_ROOT,
        )
        predictor.set_property_list(target_properties)
        predictor.run_pipeline()

        print(f"新数据预测完成！")
        print(f"预测结果JSON保存至: {pc.NEW_DATA_PREDICT_RESULT_JSON.absolute()}")
        print(f"预测结果JSON保存至: {pc.JSON_RESULT_PREDICT.absolute()}")
        print("=" * 70)
        return True

    def _generate_visualizations(self, target_properties: List[str], test_pred_df=None, test_actual_df=None, test_ids_per_prop=None):
        """生成可视化图表(训练-测试对比+簇状条形图)"""
        from .pred_actual_plots import (
            plot_pred_actual_with_train_test,
            plot_clustered_bar_each_property,
        )
        plot_pred_actual_with_train_test(
            train_pred_file="",
            test_pred_file="",
            test_pred_sheet="预测结果",
            test_actual_file="",
            test_actual_sheet="Targets",
            output_img_prefix=pc.TRAIN_TEST_COMPARE_PLOT,
            property_list=target_properties,
            best_models_dir=pc.BEST_MODEL_ROOT,
            figsize_scatter=(15, 10),
            figsize_bar=(18, 6),
            cols_per_row=2,
            scale_factor=1.8,
            random_seed=self.config.random_seed,
            metrics_json_dir=pc.JSON_RESULT_TRAIN_ROOT,
            train_data_dict=self._build_train_data_dict(),
            test_data_dict=test_pred_df,
            test_actual_df=test_actual_df,
            outlier_n=self.config.outlier_n,
            test_ids_per_prop=test_ids_per_prop,
        )

        plot_clustered_bar_each_property(
            train_pred_file="",
            test_pred_file="",
            test_pred_sheet="预测结果",
            test_actual_file="",
            test_actual_sheet="Targets",
            output_img_prefix=pc.CLUSTERED_BAR_PLOT,
            property_list=target_properties,
            max_samples_per_plot=100,
            figsize=(18, 8),
            train_data_dict=self._build_train_data_dict(),
            test_data_dict=test_pred_df,
            test_actual_df=test_actual_df,
            test_ids_per_prop=test_ids_per_prop,
        )
        print(f"可视化图表已保存至: {os.path.dirname(pc.TRAIN_TEST_COMPARE_PLOT)}")

    def _build_train_data_dict(self):
        """从best_model_perform.json构建训练数据dict供可视化使用"""
        import json
        perform_path = str(pc.BEST_MODEL_PERFORM_JSON)
        if os.path.exists(perform_path):
            with open(perform_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _backfill_test_metrics_to_best_models(self):
        """评估完成后，将最终测试集 R²/MAPE 写回 best_model_info_{prop}.json 的
        test_r2/test_mape 字段，供预测输出 model_info 展示模型泛化能力。

        （best_model_info 中原有的 cv_avg_r2/cv_avg_mape 是特征选择阶段的交叉验证
        指标，与最终测试集评估结果不同，预测端需要真正的 test 指标。）
        """
        import json as _json
        test_json = pc.JSON_RESULT_TRAIN_ROOT / "output_test.json"
        if not test_json.exists():
            print("回写测试集指标：未找到 output_test.json，跳过")
            return
        try:
            with open(test_json, "r", encoding="utf-8") as f:
                test_data = _json.load(f)
        except Exception as e:
            print(f"回写测试集指标：读取 output_test.json 失败: {e}")
            return

        for prop, entry in test_data.items():
            if not isinstance(entry, dict):
                continue
            r2 = entry.get("r2")
            mape = entry.get("mape")
            if r2 is None and mape is None:
                continue
            safe_prop = prop.replace("（", "_").replace("）", "_").replace("/", "_")
            info_file = os.path.join(str(pc.BEST_MODEL_ROOT), f"best_model_info_{safe_prop}.json")
            if not os.path.exists(info_file):
                print(f"回写测试集指标：{prop} 无 best_model_info 文件，跳过")
                continue
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    model_info = _json.load(f)
                model_info["test_r2"] = r2
                model_info["test_mape"] = mape
                with open(info_file, "w", encoding="utf-8") as f:
                    _json.dump(model_info, f, ensure_ascii=False, indent=2)
                print(f"回写测试集指标：{prop} test_r2={r2}, test_mape={mape}")
            except Exception as e:
                print(f"回写测试集指标：{prop} 写入失败: {e}")


    def get_prop_list_from_json(self, dir_path: str = '.') -> list:
        """读取目录下所有best_features_{prop}.json的prop并返回列表"""
        return [
            f.replace('best_features_', '').replace('.json', '')
            for f in os.listdir(dir_path)
            if f.startswith('best_features_') and f.endswith('.json') and os.path.isfile(os.path.join(dir_path, f))
        ]   


    def get_prop_list_from_json_inc(self, dir_path: str = '.') -> list:
        """读取目录下所有inc_{prop}.json的prop并返回列表"""
        return [
            f.replace('inc_', '').replace('.json', '')
            for f in os.listdir(dir_path)
            if f.startswith('inc_') and f.endswith('.json') and os.path.isfile(os.path.join(dir_path, f))
        ]   

    def run_pipeline(self, pipeline_type: str):
        """
        运行核心管道,新增支持独立预测/评估模式
        :param pipeline_type: 管道类型,可选：train_test_full/train_test_inc/predict/evaluate
        """
        print("=" * 60)
        print(f"启动通用机器学习预测管道 | 配置核数: {self.config.n_jobs}")
        print(f"当前运行模式: {pipeline_type} | Test_size: {self.config.test_size}")
        print("=" * 60)
        
        self.set_global_n_jobs()
        # 扩展支持的管道类型：原有训练模式 + 新增独立预测/评估
        valid_pipeline_types = ['train_full', 'train_inc', 'predict', 'evaluate']
        if pipeline_type not in valid_pipeline_types:
            raise ValueError(
                f"支持的运行模式：{valid_pipeline_types},当前传入: {pipeline_type}"
            )
        target_properties = [self.config.properties[i] for i in self.config.property_indices]
        # 执行对应管道逻辑(test_size=0时自动跳过评估)
        try:
            if pipeline_type == 'train_full':
                # 全量模型训练 + 可选评估
                train_results = self.train_full_model()
                if self.config.test_size == 0:
                    _enable_figures_base64 = bool(
                        getattr(self.config, 'enable_figures_base64', False)
                        or getattr(getattr(self, '_optimizer', None), 'enable_figures_base64', False)
                    )
                    copy_best_model_jsons(target_properties,pc.BEST_MODEL_ROOT,pc.JSON_RESULT_TRAIN_ROOT)
                    merge(str(pc.JSON_RESULT_TRAIN_ROOT), "output_train.json")
                    # 项点更新：仅更新选中的项点，其他项点及顶层键正常保留
                    merge_flat_json_update(
                        str(pc.JSON_RESULT_TRAIN_ROOT / "output_train.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "output.json"),
                        updated_props=target_properties,
                        enable_figures_base64=_enable_figures_base64,
                    )
                    file_path = pc.JSON_RESULT_TRAIN_ROOT / "output.json"
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)                        
                    except (FileNotFoundError, json.JSONDecodeError):
                        existing_data = {"properties_total": []} 
                    model_list = self.get_prop_list_from_json(str(pc.BEST_MODEL_ROOT))
                    new_model_data = model_list
                    existing_data["properties_total"] = new_model_data
                    with open(str(file_path), 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)   
                    
                    
                    return {"train_results": train_results}
                else:
                    eval_result = self.evaluate_model()
                    copy_best_model_jsons(target_properties,pc.BEST_MODEL_ROOT,pc.JSON_RESULT_TRAIN_ROOT)
                    merge(str(pc.JSON_RESULT_TRAIN_ROOT), "output_train.json")
                    _enable_figures_base64 = bool(
                        getattr(self.config, 'enable_figures_base64', False)
                        or getattr(getattr(self, '_optimizer', None), 'enable_figures_base64', False)
                    )
                    # 项点更新：完整评估后仅更新选中的项点，其他项点保留原记录
                    merge_train_test_json_update(
                        str(pc.JSON_RESULT_TRAIN_ROOT / "output_test.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "output_train.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "output.json"),
                        updated_props=target_properties,
                        enable_figures_base64=_enable_figures_base64,
                    )
                    if _enable_figures_base64:
                        img2base64_to_json(target_properties, str(pc.VISUAL_ROOT), str(pc.JSON_RESULT_TRAIN_ROOT / "figures.json"))
                    _generate_shap_optional(
                        enable_shap=self.config.enable_shap,
                        model_dir=str(pc.BEST_MODEL_ROOT),
                        property_list=target_properties,
                        train_features_df=self.full_features_df,
                        training_params_dir=str(pc.TRAIN_PARAMS_ROOT),
                        shap_output_dir=str(pc.SHAP_ANALYSIS),
                        figures_json_path=str(pc.JSON_RESULT_TRAIN_ROOT / "figures.json") if _enable_figures_base64 else None,
                    )
                    # 统一图表数据：合并 corr_data/pred_actual_data/shap_data 为 chart_data.json（供前端直接作图）
                    merge_chart_data_json(
                        str(pc.JSON_RESULT_TRAIN_ROOT / "corr_data.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "pred_actual_data.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "shap_data.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "chart_data.json"),
                    )
                    model_list = self.get_prop_list_from_json(str(pc.BEST_MODEL_ROOT))
                    file_path = pc.JSON_RESULT_TRAIN_ROOT / "output.json"
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        existing_data = {"properties_total": []} 
                    new_model_data = model_list
                    existing_data["properties_total"] = new_model_data
                    with open(str(file_path), 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)   
                    return {"train_results": train_results, "eval_result": eval_result}
            elif pipeline_type == 'train_inc':
                # 增量模型训练 + 可选评估
                train_results = self.train_incremental_model()
                if self.config.test_size == 0:
                    _enable_figures_base64 = bool(
                        getattr(self.config, 'enable_figures_base64', False)
                        or getattr(getattr(self, '_optimizer', None), 'enable_figures_base64', False)
                    )
                    copy_best_model_jsons(target_properties,pc.BEST_MODEL_ROOT,pc.JSON_RESULT_TRAIN_ROOT)
                    merge(str(pc.JSON_RESULT_TRAIN_ROOT), "output_train.json")
                    # 项点更新：仅更新选中的项点，其他项点及顶层键正常保留
                    merge_flat_json_update(
                        str(pc.JSON_RESULT_TRAIN_ROOT / "output_train.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "output.json"),
                        updated_props=target_properties,
                        enable_figures_base64=_enable_figures_base64,
                    )
                    model_list = self.get_prop_list_from_json(str(pc.BEST_MODEL_ROOT))
                    file_path = pc.JSON_RESULT_TRAIN_ROOT / "output.json"
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        existing_data = {"properties_total": []} 
                    new_model_data = model_list
                    existing_data["properties_total"] = new_model_data
                    with open(str(file_path), 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2) 
                    return {"train_results": train_results}
                else:
                    eval_result = self.evaluate_model()
                    copy_best_model_jsons(target_properties,pc.BEST_MODEL_ROOT,pc.JSON_RESULT_TRAIN_ROOT)
                    merge(str(pc.JSON_RESULT_TRAIN_ROOT), "output_train.json")
                    _enable_figures_base64 = bool(
                        getattr(self.config, 'enable_figures_base64', False)
                        or getattr(getattr(self, '_optimizer', None), 'enable_figures_base64', False)
                    )
                    # 项点更新：完整评估后仅更新选中的项点，其他项点保留原记录
                    merge_train_test_json_update(
                        str(pc.JSON_RESULT_TRAIN_ROOT / "output_test.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "output_train.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "output.json"),
                        updated_props=target_properties,
                        enable_figures_base64=_enable_figures_base64,
                    )
                    if _enable_figures_base64:
                        img2base64_to_json(target_properties, str(pc.VISUAL_ROOT), str(pc.JSON_RESULT_TRAIN_ROOT / "figures.json"))
                    _generate_shap_optional(
                        enable_shap=self.config.enable_shap,
                        model_dir=str(pc.BEST_MODEL_ROOT),
                        property_list=target_properties,
                        train_features_df=self.full_features_df,
                        training_params_dir=str(pc.TRAIN_PARAMS_ROOT),
                        shap_output_dir=str(pc.SHAP_ANALYSIS),
                        figures_json_path=str(pc.JSON_RESULT_TRAIN_ROOT / "figures.json") if _enable_figures_base64 else None,
                    )
                    # 统一图表数据：合并 corr_data/pred_actual_data/shap_data 为 chart_data.json（供前端直接作图）
                    merge_chart_data_json(
                        str(pc.JSON_RESULT_TRAIN_ROOT / "corr_data.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "pred_actual_data.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "shap_data.json"),
                        str(pc.JSON_RESULT_TRAIN_ROOT / "chart_data.json"),
                    )
                    model_list = self.get_prop_list_from_json(str(pc.BEST_MODEL_ROOT))
                    file_path = pc.JSON_RESULT_TRAIN_ROOT / "output.json"
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        existing_data = {"properties_total": []} 

                    new_model_data = model_list 
                    existing_data["properties_total"] = new_model_data
                    with open(str(file_path), 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)
                    return {"train_results": train_results, "eval_result": eval_result}
            elif pipeline_type == 'predict':
                # 独立预测模式：直接对新数据预测,无需训练
                pred_result = self.predict_new_data()
                return {"pred_result": pred_result}
            elif pipeline_type == 'evaluate':
                # 独立评估模式：基于已有模型和测试集评估,无需训练
                eval_result = self.evaluate_model()
                copy_best_model_jsons(target_properties,pc.BEST_MODEL_ROOT,pc.JSON_RESULT_TRAIN_ROOT)
                merge(str(pc.JSON_RESULT_TRAIN_ROOT),"output_train.json")
                _enable_figures_base64 = bool(
                    getattr(self.config, 'enable_figures_base64', False)
                    or getattr(getattr(self, '_optimizer', None), 'enable_figures_base64', False)
                )
                # 项点更新：完整评估后仅更新选中的项点，其他项点保留原记录
                merge_train_test_json_update(
                    str(pc.JSON_RESULT_TRAIN_ROOT / "output_test.json"),
                    str(pc.JSON_RESULT_TRAIN_ROOT / "output_train.json"),
                    str(pc.JSON_RESULT_TRAIN_ROOT / "output.json"),
                    updated_props=target_properties,
                    enable_figures_base64=_enable_figures_base64,
                )
                if _enable_figures_base64:
                    img2base64_to_json(target_properties, str(pc.VISUAL_ROOT), str(pc.JSON_RESULT_TRAIN_ROOT / "figures.json"))
                # 统一图表数据：合并 corr_data/pred_actual_data/shap_data 为 chart_data.json（供前端直接作图）
                merge_chart_data_json(
                    str(pc.JSON_RESULT_TRAIN_ROOT / "corr_data.json"),
                    str(pc.JSON_RESULT_TRAIN_ROOT / "pred_actual_data.json"),
                    str(pc.JSON_RESULT_TRAIN_ROOT / "shap_data.json"),
                    str(pc.JSON_RESULT_TRAIN_ROOT / "chart_data.json"),
                )
                return {"eval_result": eval_result}
        except Exception as e:
            print(f"管道执行失败,错误信息: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise

def main(
    # 1. 核心配置参数
    properties=None,
    property_indices=None,
    model_type='custom',
    test_size=0.0,
    n_iterations=3,
    n_folds=4,
    random_seed=100,
    judge=0,
    max_no_improve_rounds=0,
    search_method='random',
    nn_backend='sklearn',
    n_jobs=16,
    model_list_custom=None,
    reduction_type='custom',
    mode_list=None,
    # 2. 调试参数
    debug_dump_csv=False,
    outlier_n=None,
    # 3. 异常样本剔除参数
    remove_outliers=False,
    outlier_std_threshold=2.0,
    outlier_model='ridge',
    # 4. SHAP 可解释性开关
    enable_shap=True,
    # 5. 管道运行模式参数
    pipeline_type='predict'
):
    # ==================== 1. 核心配置(使用传参/默认值)===================
    config = PipelineConfig(
        properties=properties,
        property_indices=property_indices,
        model_type=model_type,
        test_size=test_size,
        n_iterations=n_iterations,
        n_folds=n_folds,
        random_seed=random_seed,
        judge=judge,
        max_no_improve_rounds=max_no_improve_rounds,
        search_method=search_method,
        nn_backend=nn_backend,
        n_jobs=n_jobs,
        model_list_custom=model_list_custom,
        reduction_type=reduction_type,
        mode_list=mode_list,
        debug_dump_csv=debug_dump_csv,
        outlier_n=outlier_n,
        remove_outliers=remove_outliers,
        outlier_std_threshold=outlier_std_threshold,
        outlier_model=outlier_model,
        enable_shap=enable_shap,
    )

    # ==================== 2. 路径初始化(使用传参/默认值)==================
    if pipeline_type != 'train_full':
        pc.init_paths()

        import glob
        new_data_files = glob.glob(str(pc.OUTPUT_ROOT / "new_data_processed*"))
        for f in new_data_files:
            os.remove(f)
            print(f"已清除旧文件: {f}")
        for pred_file in [pc.TEST_PREDICT_RESULT_JSON, pc.NEW_DATA_PREDICT_RESULT_JSON]:
            if pred_file.exists():
                os.remove(str(pred_file))
                print(f"已清除旧文件: {pred_file}")

    # ==================== 3. 创建并运行管道(使用传参/默认值)===================
    pipeline = GenericMLPipeline(config)
    pipeline.run_pipeline(pipeline_type=pipeline_type)

    print("=" * 60)
    print("通用机器学习预测管道执行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
