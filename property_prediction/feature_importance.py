# -*- coding: utf-8 -*-
"""
低开销特征重要性计算模块（不依赖SHAP）
按模型类型分派：
- 线性类模型（ridge/lasso/elasticnet/bayesian_ridge/linear/huber/linearsvr）：
    对标准化后的X重新拟合取 |coef_|，消除特征量纲影响，保证重要性可比
- 树类模型（dt/rf/extra_trees/gbr/gbdt/hist_gbdt/adaboost/xgb/lgbm/catboost）：
    直接取 feature_importances_（训练时已算好，零额外成本）
- Poly管线（PolynomialFeatures + LinearRegression）：
    多项式系数按原始特征聚合（含该变量的所有单项式系数平方和开方），
    避免permutation在多项式特征爆炸时（如 degree=3 × 268特征 ≈ 328万列）卡死
- PyTorch神经网络（fnn/deep_fnn/simple_fnn/resnet，PyTorchTrainer）：
    集成梯度（Integrated Gradients），基于特征的均值基线沿插值路径累加输入梯度，
    一次前向+反向即可，比permutation便宜
- 其余模型（svr/svr_rbf/knn/gpr/sklearn-MLP等）：
    sklearn permutation_importance（模型无关，成本 = n_repeats * n_features * 单次预测）；
    高维特征(>60)时自动降为3次重复并开启并行，避免长时间卡住
"""
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

TREE_MODELS = {'dt', 'rf', 'extra_trees', 'gbr', 'gbdt', 'hist_gbdt',
               'adaboost', 'xgb', 'lgbm', 'catboost'}
LINEAR_MODELS = {'linear', 'ridge', 'lasso', 'elasticnet',
                 'bayesian_ridge', 'huber', 'linearsvr'}
NN_MODELS = {'fnn', 'deep_fnn', 'simple_fnn', 'resnet'}


def _unwrap_estimator(model):
    """解包Pipeline，取出最终estimator"""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    if hasattr(model, 'named_steps'):
        vals = list(model.named_steps.values())
        return vals[-1] if vals else model
    return model


def _is_pytorch_trainer(model) -> bool:
    """判断是否为PyTorchTrainer（包装了nn.Module）"""
    try:
        import torch
    except ImportError:
        return False
    return hasattr(model, 'model') and isinstance(getattr(model, 'model'), torch.nn.Module)


def _pytorch_integrated_gradients(model, X: pd.DataFrame, n_steps: int = 10) -> np.ndarray:
    """集成梯度：以特征均值为基线，沿直线插值到每个样本并累加输入梯度。

    返回形状为 (n_features,) 的逐特征归因（每个样本的 (x-baseline)*梯度 之和 / 样本数 / 步数）。
    """
    import torch
    nn_model = model.model
    device = getattr(model, 'device', torch.device('cpu'))
    nn_model.eval()

    X_np = np.asarray(X.values, dtype=np.float32)
    baseline_np = np.mean(X_np, axis=0, keepdims=True)
    X_t = torch.FloatTensor(X_np).to(device)
    baseline_t = torch.FloatTensor(baseline_np).to(device)
    n = len(X_np)

    total = torch.zeros(X_t.shape[1], dtype=torch.float32, device=device)
    for alpha in np.linspace(0.0, 1.0, n_steps):
        interp = baseline_t + (X_t - baseline_t) * float(alpha)
        interp.requires_grad_(True)
        nn_model.zero_grad()
        out = nn_model(interp)
        out.sum().backward()
        grad = interp.grad  # (n, n_features)
        total += (grad * (X_t - baseline_t)).sum(dim=0)

    ig = total / n_steps / n
    return ig.cpu().numpy()


def _poly_aggregated_importance(model, X: pd.DataFrame) -> np.ndarray:
    """Poly管线：把多项式系数按原始特征聚合。

    对每个原始特征，累加所有包含该变量的单项式系数平方后开方，
    得到各原始特征的整体重要性，避免对爆炸后的多项式特征做permutation。
    """
    if not isinstance(model, Pipeline) or not hasattr(model, 'named_steps'):
        raise ValueError("非Pipeline结构，无法按poly聚合")
    if 'poly' not in model.named_steps:
        raise ValueError("管线中无poly步骤")
    poly = model.named_steps['poly']
    if not hasattr(poly, 'get_feature_names_out'):
        raise ValueError("PolynomialFeatures版本过低，缺少get_feature_names_out")

    estimator = _unwrap_estimator(model)
    names = list(poly.get_feature_names_out(X.columns))
    coef = np.asarray(estimator.coef_).ravel()
    if len(coef) != len(names):
        raise ValueError(f"系数长度({len(coef)})与多项式特征数({len(names)})不一致")

    col_index = {c: i for i, c in enumerate(X.columns)}
    agg = np.zeros(len(X.columns))
    for k, name in enumerate(names):
        for token in name.split(' '):
            base = token.split('^')[0]  # 去掉幂次，如 'f0^2' -> 'f0'
            if base in col_index:
                agg[col_index[base]] += coef[k] ** 2
    return np.sqrt(agg)


def _permutation_imp(model, X, y, n_repeats, n_jobs, random_state):
    result = permutation_importance(model, X, y, scoring='r2',
                                    n_repeats=n_repeats, n_jobs=n_jobs,
                                    random_state=random_state)
    return result.importances_mean


def _permutation_imp_subset(model, X: pd.DataFrame, y: pd.Series, feature_names: list,
                            n_repeats: int, n_jobs: int, random_state: int) -> np.ndarray:
    """只对指定列做置换的permutation importance（sklearn>=1.1无col_to_permute，手动实现）。

    置换逐列独立：计算某一列时其余列保持真实值，因此结果与对全列置换完全一致，
    只是跳过其余列（如降维主成分）的置换开销。随机流与sklearn 1.8一致：
    由random_state抽取一个随机种子，各列用该种子新建独立RandomState，
    并在同一副本上跨repeat原地累积打乱，保证与sklearn全量计算结果逐位一致。
    """
    from joblib import Parallel, delayed
    from sklearn.metrics import get_scorer

    scorer = get_scorer('r2')
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X_np = np.asarray(X)
    baseline = float(scorer(model, X, y))
    # 与sklearn._permutation_importance相同的种子派生方式
    random_seed = int(np.random.RandomState(random_state).randint(np.iinfo(np.int32).max + 1))
    col_idx = [i for i, c in enumerate(X.columns) if c in set(feature_names)]

    def _one(ci):
        col_rng = np.random.RandomState(random_seed)
        X_perm = X_np.copy()
        shuffling_idx = np.arange(X_np.shape[0])
        drops = np.empty(n_repeats)
        for n in range(n_repeats):
            col_rng.shuffle(shuffling_idx)
            X_perm[:, ci] = X_perm[shuffling_idx, ci]  # 只打乱该列，其余列保持真实值
            drops[n] = baseline - float(scorer(model, X_perm, y))
        return drops.mean()

    if n_jobs == 1 or n_jobs is None:
        vals = [_one(ci) for ci in col_idx]
    else:
        vals = Parallel(n_jobs=n_jobs)(delayed(_one)(ci) for ci in col_idx)
    return np.asarray(vals)


def compute_feature_importance(model, X: pd.DataFrame, y: pd.Series, model_type: str,
                               n_repeats: int = 5, n_jobs: int = 1,
                               random_state: int = 42, n_steps: int = 10,
                               features: Optional[list] = None) -> pd.Series:
    """计算各特征的重要性，按 X 的列顺序返回，已归一化到和为1。

    features: 可选的特征子集（列名列表）。指定后只返回这些特征的重要性：
    模型仍在完整输入空间上训练/预测（保证数值正确），permutation 只对子集列置换，
    跳过其余列（如降维主成分）的开销；线性coef/树fi/poly/集成梯度在完整空间计算后切片。
    >60特征降重复次数的判据也基于子集特征数。不指定时等价于对全部列计算。

    若所有方法均无法计算（如PyTorch模型predict接口不兼容），抛出异常，由调用方降级处理。
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    estimator = _unwrap_estimator(model)

    # 子集列索引（保持 X 的列顺序）
    if features is not None:
        feat_set = set(features)
        subset_idx = [i for i, c in enumerate(X.columns) if c in feat_set]
        subset_cols = [X.columns[i] for i in subset_idx]
    else:
        subset_idx = list(range(X.shape[1]))
        subset_cols = list(X.columns)
    n_subset = len(subset_idx)

    imp = None

    # 0) PyTorch神经网络：集成梯度（比permutation便宜）
    if model_type in NN_MODELS and _is_pytorch_trainer(model):
        try:
            ig = _pytorch_integrated_gradients(model, X, n_steps=n_steps)
            if len(ig) == X.shape[1]:
                imp = np.abs(ig)
        except Exception as e:
            print(f"  [importance] PyTorch集成梯度计算失败: {e}，改用permutation")
            imp = None

    # 1) 线性模型：标准化后重拟合取 |coef_|（量纲可比）
    if imp is None and model_type in LINEAR_MODELS and hasattr(estimator, 'coef_'):
        try:
            X_std = StandardScaler().fit_transform(X)
            est = clone(estimator)
            est.fit(X_std, y)
            coef = np.asarray(est.coef_).ravel()
            if len(coef) == X.shape[1]:
                imp = np.abs(coef)
        except Exception as e:
            print(f"  [importance] 线性coef计算失败: {e}，改用permutation")
            imp = None

    # 2) 树模型：直接取 feature_importances_
    if imp is None and model_type in TREE_MODELS and hasattr(estimator, 'feature_importances_'):
        fi = np.asarray(estimator.feature_importances_).ravel()
        if len(fi) == X.shape[1]:
            imp = fi

    # 2.5) Poly管线：多项式系数按原始特征聚合（避免permutation在特征爆炸时卡死）
    if imp is None and model_type == 'poly':
        try:
            imp = _poly_aggregated_importance(model, X)
        except Exception as e:
            print(f"  [importance] poly系数聚合失败: {e}，改用permutation")
            imp = None

    # 3) 兜底：permutation importance（模型无关；指定子集时只置换子集列）
    if imp is None:
        eff_repeats = n_repeats
        eff_jobs = n_jobs
        if n_subset > 60:
            # 参与置换的特征数高时permutation开销大，降重复次数并开并行，避免长时间卡住
            eff_repeats = min(n_repeats, 3)
            eff_jobs = -1 if n_jobs <= 1 else n_jobs
            print(f"  [importance] 参与置换的特征数={n_subset}>60，permutation降为{eff_repeats}次重复并开启并行")
        if features is not None:
            imp = _permutation_imp_subset(model, X, y, subset_cols, eff_repeats, eff_jobs, random_state)
        else:
            imp = _permutation_imp(model, X, y, eff_repeats, eff_jobs, random_state)

    # 统一按子集切片（全量计算时切片退化；permutation子集路径已是子集长度）
    imp_arr = np.asarray(imp).ravel()
    if features is not None:
        if len(imp_arr) == X.shape[1]:
            imp_arr = imp_arr[subset_idx]
        out_cols = subset_cols
    else:
        out_cols = list(X.columns)

    series = pd.Series(imp_arr, index=out_cols, dtype=float)
    total = series.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("特征重要性总和为0，无法归一化")
    return series / total
