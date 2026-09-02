from pathlib import Path


# ======================== 模型输入 ========================

# 输入路径 - JSON格式
TRAIN_JSON = Path("./runtime/property_prediction/train.json")
PREDICT_JSON = Path("./runtime/property_prediction/predict.json")

# ======================== 根路径 ========================
OUTPUT_ROOT = Path("./artifacts/property_prediction")
VISUAL_ROOT = Path("./results/property_prediction/visualizations")
JSON_RESULT_ROOT = Path("./results/property_prediction")

# ======================== 中间文件 ========================
# 核心功能目录（基于OUTPUT_ROOT）
TRAIN_PARAMS_ROOT = OUTPUT_ROOT / "reduction_params"  # 降维参数保存目录, 预测评估与增量训练需读取该目录
BEST_MODEL_ROOT = OUTPUT_ROOT / "best_models"  # 最优模型保存目录，预测评估与增量训练需读取该目录
INCREMENTAL_OUTPUT_ROOT = OUTPUT_ROOT  # 增量训练输出目录
FINAL_MODEL_ROOT = OUTPUT_ROOT / "final_models_params"  # 全量数据最终模型保存目录，预测时优先读取（standard/降维params/pkl）

# 中间生成文件 - 数据处理后（JSON格式）
PROCESSED_TRAIN_JSON = OUTPUT_ROOT / "new_data_processed.json"  # 训练集处理后JSON
PROCESSED_PREDICT_JSON = OUTPUT_ROOT / "new_data_processed_predict.json"  # 预测集处理后JSON
# 预测结果 - JSON格式
TEST_PREDICT_RESULT_JSON = OUTPUT_ROOT / "新数据性质预测结果_测试模式.json"  # 测试模式预测结果JSON
NEW_DATA_PREDICT_RESULT_JSON = OUTPUT_ROOT / "新数据性质预测结果.json"  # 新数据预测结果JSON
# 模型性能与标准化参数
BEST_MODEL_PERFORM_JSON = OUTPUT_ROOT / "best_model_perform.json"  # 最优模型性能JSON
STANDARDIZATION_PARAMS = OUTPUT_ROOT / "standardization_params_.pkl"  # 特征标准化参数文件

# ======================== 模型输出 ========================
# 可视化文件前缀
TRAIN_TEST_COMPARE_PLOT = VISUAL_ROOT / "训练测试集对比"  # 训练测试集对比图前缀
CLUSTERED_BAR_PLOT = VISUAL_ROOT / "簇状条形图"  # 簇状条形图文件前缀
CORR_BAR_PLOT = VISUAL_ROOT / "相关性分析"  # 簇状条形图文件前缀
SHAP_ANALYSIS = VISUAL_ROOT / "SHAP分析"  # SHAP可解释性分析图目录
# 训练预测推荐结果目录
JSON_RESULT_TRAIN_ROOT = JSON_RESULT_ROOT / "training"
JSON_RESULT_PREDICT_ROOT = JSON_RESULT_ROOT / "prediction"

# 预测推荐结果文件
JSON_RESULT_PREDICT = JSON_RESULT_PREDICT_ROOT / "output.json" # 测试相关JSON输出目录

# ======================== 路径初始化函数 ========================
def init_paths():
    """初始化路径：自动创建所有所需目录"""
    dirs_to_create = [
        OUTPUT_ROOT, VISUAL_ROOT, JSON_RESULT_ROOT, CORR_BAR_PLOT,
        TRAIN_PARAMS_ROOT, BEST_MODEL_ROOT, INCREMENTAL_OUTPUT_ROOT,
        FINAL_MODEL_ROOT,
        JSON_RESULT_TRAIN_ROOT, JSON_RESULT_PREDICT_ROOT, SHAP_ANALYSIS
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"路径初始化：创建目录 {dir_path.absolute()}")
