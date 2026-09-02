import os
import json
import shutil
from pathlib import Path
from typing import List, Optional
import base64


# merge 时保留、不参与逐项点合并/删除的JSON文件名
RESERVED_JSON_NAMES = {
    "output_train.json", "output_test.json", "output.json", "figures.json",
    # 统一图表数据文件（由 corr_data/pred_actual_data/shap_data 合并而来，供前端作图）
    "chart_data.json",
    # 图表数据中间文件：无 property 键，merge() 须跳过；流程末尾由 merge_chart_data_json 合并进 chart_data.json 后删除
    "corr_data.json", "shap_data.json", "pred_actual_data.json",
}


def img2base64_to_json(properties_list: list, path1: str, path2: str):
    """
    检索指定路径图片转base64,按指定结构生成json并保存（全部base64仅写入figures.json，不与output.json合并）
    增量保留：figures.json 已存在时，本轮未重训性质的原base64保留，仅覆盖/更新本轮选中的性质
    :param properties_list: 目标属性列表(prop1/prop2...)
    :param path1: 源图片根路径(含3个文件夹+1张结论图)
    :param path2: 最终json输出路径(含文件名,如./output/result.json)
    :return: 无
    """
    # 定义固定文件夹名、图片后缀(兼容常见格式)
    FOLDERS = ["簇状条形图_bar", "训练测试集对比_scatter", "相关性分析"]
    IMG_SUFFIX = (".png", ".jpg", ".jpeg", ".bmp")
    result = {}  # 最终结果字典
    # 增量保留：读取已有 figures.json，未在本轮重训的性质 base64 原样保留
    if os.path.exists(path2):
        try:
            with open(path2, "r", encoding="utf-8") as f:
                _existing = json.load(f)
            if isinstance(_existing, dict):
                result = _existing
        except (json.JSONDecodeError, OSError):
            result = {}

    # 1. 遍历每个prop,检索3个文件夹下的对应图片（本轮选中的性质整体覆盖为最新base64）
    for prop in properties_list:
        result[prop] = {}
        for folder in FOLDERS:
            folder_path = os.path.join(path1, folder)
            if not os.path.exists(folder_path):
                result[prop][folder] = ""
                continue
            # 遍历文件夹找prop开头的图片
            for file in os.listdir(folder_path):
                if file.startswith(prop) and file.lower().endswith(IMG_SUFFIX):
                    file_path = os.path.join(folder_path, file)
                    # 图片转base64(二进制读取+编码+转字符串,适配json存储)
                    with open(file_path, "rb") as f:
                        b64_str = base64.b64encode(f.read()).decode("utf-8")
                    result[prop][folder] = b64_str
                    break  # 假定每个prop对应一个图片,找到即退出
            else:
                result[prop][folder] = ""  # 无对应图片时设为空

    # 2. 处理根目录的conclusion图片（未找到新结论图时保留旧结论图base64）
    _conclusion_found = False
    for file in os.listdir(path1):
        file_path = os.path.join(path1, file)
        if os.path.isfile(file_path) and file.startswith("训练测试集对比") and file.lower().endswith(IMG_SUFFIX):
            with open(file_path, "rb") as f:
                result["conclusion"] = base64.b64encode(f.read()).decode("utf-8")
            _conclusion_found = True
            break
    if not _conclusion_found:
        result.setdefault("conclusion", "")
    # 3. 保存图片base64原始json文件（所有base64均只写入figures.json，不再与output.json合并）
    with open(path2, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"图片base64 json文件已生成：{path2}")


# 需要从合并结果中剔除的base64图片键（enable_figures_base64=False 时清理历史残留）
BASE64_KEYS = (
    "conclusion",         # 顶层：结论对比图
    "簇状条形图_bar",     # 各性质：图片base64
    "训练测试集对比_scatter",
    "相关性分析",
    "SHAP_summary",       # 各性质：SHAP图片base64
    "SHAP_waterfall",
    "SHAP_dependence",
)


def _strip_base64_keys(data: dict) -> None:
    """从output.json合并结果中剔除base64图片键（含顶层conclusion与各性质的图片键）。

    同名键同时承载 base64（字符串）与图表数据（dict，来自chart_data.json合并），
    因此仅剔除字符串类型的base64值，保留dict图表数据。
    """
    conclusion_val = data.get("conclusion")
    if isinstance(conclusion_val, str):
        data.pop("conclusion", None)
    for prop in list(data.keys()):
        node = data[prop]
        if isinstance(node, dict):
            for k in BASE64_KEYS:
                v = node.get(k)
                if not isinstance(v, dict):
                    node.pop(k, None)


def merge(json_dir, out_file):
    json_dir = json_dir
    # r"results/property_prediction/training"  # JSON结果目录
    output_file = out_file     # 合并后的输出文件名,默认同目录


    merged_data = {}

    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json") and file_name not in RESERVED_JSON_NAMES:
            file_path = os.path.join(json_dir, file_name)  
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = json.load(f)
            prop_key = file_content["property"]
            merged_data[prop_key] = file_content

    if "train" not in output_file and "test" not in output_file:
        _add_fit_status_unwrapped(merged_data)

    with open(os.path.join(json_dir, output_file), "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"合并完成！共处理 {len(merged_data)} 个JSON文件,输出文件：{os.path.join(json_dir, output_file)}")

    for file_name in os.listdir(json_dir):
        file_path = os.path.join(json_dir, file_name)
        if file_name not in RESERVED_JSON_NAMES:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"已删除：{file_name}")
            except Exception as e:
                print(f"删除文件{file_name}失败,错误信息：{str(e)}")

def merge_train_test_json(
    test_json: Path = Path("output_test.json"),
    train_json: Path = Path("output_train.json"),
    output_json: Path = Path("output.json")
):
    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    merged_data = {}
    for prop_name in train_data.keys():
        merged_data[prop_name] = {
            "train": train_data[prop_name],
            "test": test_data.get(prop_name, {})
        }
    for prop_name in test_data.keys():
        if prop_name not in merged_data:
            merged_data[prop_name] = {"train": {}, "test": test_data[prop_name]}

    _add_fit_status(merged_data)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"合并成功")

    return merged_data


def merge_train_test_json_update(
    test_json: Path = Path("output_test.json"),
    train_json: Path = Path("output_train.json"),
    output_json: Path = Path("output.json"),
    updated_props: Optional[List[str]] = None,
    enable_figures_base64: bool = True,
):
    """项点更新合并（train+test 结构，全量/增量训练均使用）：
    - updated_props（选中的项点）：用本次训练/评估的 train 与 test 记录更新
    - 其他项点：原样保留 output.json 中已有的记录（缺失时用本次结果补齐）
    - 已存在的其他顶层结果区块一律保留
    """
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    output_json = Path(output_json)
    existing_data = {}
    if output_json.exists():
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_data = {}
    if not isinstance(existing_data, dict):
        existing_data = {}

    updated_props = updated_props or []

    all_props = sorted(set(list(train_data.keys()) + list(test_data.keys())))
    for prop_name in all_props:
        entry = {
            "train": train_data.get(prop_name, {}),
            "test": test_data.get(prop_name, {}),
        }
        train_r2 = entry.get("train", {}).get("train_eval_metrics", {}).get("r2")
        test_r2 = entry.get("test", {}).get("r2")
        _set_fit_status(entry, train_r2, test_r2)
        # 选中的项点用新结果覆盖，其余项点仅在缺失时补齐
        if prop_name in updated_props or prop_name not in existing_data:
            existing_data[prop_name] = entry

    if not enable_figures_base64:
        _strip_base64_keys(existing_data)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"项点更新合并完成（选中项点 {len(updated_props)} 个已更新，其余保留）: {output_json}")
    return existing_data


def merge_flat_json_update(
    flat_json: Path,
    output_json: Path,
    updated_props: Optional[List[str]] = None,
    enable_figures_base64: bool = True,
):
    """项点更新合并（扁平结构，用于 test_size=0 的全量训练，无测试集）：
    flat_json 内容为 {prop: best_model_info}，output.json 仅更新选中的项点，其余保留。
    """
    with open(flat_json, 'r', encoding='utf-8') as f:
        flat_data = json.load(f)

    output_json = Path(output_json)
    existing_data = {}
    if output_json.exists():
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_data = {}
    if not isinstance(existing_data, dict):
        existing_data = {}

    updated_props = updated_props or []
    for prop_name, info in flat_data.items():
        # 选中的项点用新模型信息覆盖，其余项点仅在缺失时补齐
        if prop_name in updated_props or prop_name not in existing_data:
            entry = dict(info)
            # 无测试集时训练 R² 由 train_eval_metrics 提供，拟合状态按原结构补全
            train_r2 = entry.get("train_eval_metrics", {}).get("r2")
            _set_fit_status(entry, train_r2, None)
            existing_data[prop_name] = entry

    if not enable_figures_base64:
        _strip_base64_keys(existing_data)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"扁平项点更新合并完成（选中项点 {len(updated_props)} 个已更新，其余保留）: {output_json}")
    return existing_data


def _add_fit_status(merged_data):
    for prop_name, entry in merged_data.items():
        train_r2 = entry.get("train", {}).get("train_eval_metrics", {}).get("r2")
        test_r2 = entry.get("test", {}).get("r2")
        _set_fit_status(entry, train_r2, test_r2)


def _add_fit_status_unwrapped(merged_data):
    for prop_name, entry in merged_data.items():
        train_r2 = entry.get("train_eval_metrics", {}).get("r2")
        test_r2 = entry.get("test", {}).get("r2")
        _set_fit_status(entry, train_r2, test_r2)


def _set_fit_status(entry, train_r2, test_r2, train_threshold=0.5, gap_threshold=0.15):
    """
    根据训练集和测试集的 R² 判定模型拟合状态。

    判断逻辑(封闭且互斥)：
    1. 任一 R² 缺失                     → 未知
    2. 训练集 R² < train_threshold     → 欠拟合(连训练数据都未充分学习)
    3. train_r2 - test_r2 > gap_threshold → 过拟合(训练好但泛化差,train/test 差距过大)
    4. 其余情况                         → 正常拟合

    Parameters:
        train_threshold: 训练集 R² 最低阈值,低于此值视为欠拟合。默认 0.5。
        gap_threshold:   train_r2 - test_r2 的容忍上限,超过此值视为过拟合。默认 0.15。
    """
    if train_r2 is None or test_r2 is None:
        entry["拟合状态"] = "未知"
    elif train_r2 < train_threshold:
        entry["拟合状态"] = "欠拟合"
    elif (train_r2 - test_r2) > gap_threshold:
        entry["拟合状态"] = "过拟合"
    else:
        entry["拟合状态"] = "正常拟合"


def copy_best_model_jsons(property_list, src_dir, dst_dir):
    """
    按property列表复制指定JSON文件从目录A到目录B
    """
    # 确保目标目录存在,不存在则自动创建
    os.makedirs(dst_dir, exist_ok=True)
    # 遍历property列表执行复制
    for prop in property_list:
        # 拼接源文件和目标文件完整路径
        src_file = os.path.join(src_dir, f"best_model_info_{prop}.json")
        dst_file = os.path.join(dst_dir, f"best_model_info_{prop}.json")
        try:
            shutil.copy2(src_file, dst_file)
            print(f"成功复制：{os.path.basename(src_file)} -> {dst_dir}")
        except FileNotFoundError:
            print(f"警告：文件不存在,跳过 -> {src_file}")


# ==================== 统一图表数据文件（chart_data.json） ====================
# 替代分散的 corr_data.json / pred_actual_data.json / shap_data.json，
# 结构：{prop: {"簇状条形图_bar", "训练测试集对比_scatter", "相关性分析",
#               "SHAP_summary", "SHAP_waterfall", "SHAP_dependence"}, ...,
#        "conclusion": {"r2"/"mape"/"pearson": {prop: {"训练集": v, "测试集": v}}}}
CHART_DATA_JSON_NAME = "chart_data.json"


def _read_json_file(path) -> Optional[dict]:
    """安全读取JSON文件，不存在/损坏返回None"""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _build_conclusion_data(pred_actual_data: dict) -> dict:
    """从 pred_actual_data 的 metrics 汇总 conclusion 数据（r2/mape/pearson 三组，各含训练集+测试集）"""
    result = {"r2": {}, "mape": {}, "pearson": {}}
    for prop, entry in (pred_actual_data or {}).items():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get('metrics') or {}
        train_cn = (metrics.get('训练集') or {}).get('cn') or metrics.get('训练集') or {}
        test_cn = (metrics.get('测试集') or {}).get('cn') or metrics.get('测试集') or {}
        result["r2"][prop] = {
            "训练集": train_cn.get('R2'),
            "测试集": test_cn.get('R2'),
        }
        result["mape"][prop] = {
            "训练集": train_cn.get('MAPE'),
            "测试集": test_cn.get('MAPE'),
        }
        result["pearson"][prop] = {
            "训练集": train_cn.get('Pearson相关系数'),
            "测试集": test_cn.get('Pearson相关系数'),
        }
    return result


def _build_shap_chart_data(shap_entry: dict) -> dict:
    """把 shap_data 中单个性质的数据拆分为 SHAP_summary/waterfall/dependence 三份可直接作图的数据。

    - SHAP_summary（全局摘要 beeswarm，与 shap.summary_plot 一致）: 所有样本的 data + shap_values
    - SHAP_waterfall（单样本瀑布图）: 已挑选样本的数据
    - SHAP_dependence（TOP特征依赖图）: TOP特征全样本散点数据
    """
    if not isinstance(shap_entry, dict):
        return {}
    feature_names = shap_entry.get('feature_names') or []
    importance = shap_entry.get('importance') or {}
    base_value = shap_entry.get('base_value')
    sample_idx = shap_entry.get('waterfall_sample') or 0
    samples = shap_entry.get('samples') or {}
    data_all = samples.get('data') or []
    shap_all = samples.get('shap_values') or []

    # 瀑布图：已挑选样本的数据（sample_idx 越界时回退到第 0 条）
    w_data = data_all[sample_idx] if data_all and sample_idx < len(data_all) else (data_all[0] if data_all else [])
    w_shap = shap_all[sample_idx] if shap_all and sample_idx < len(shap_all) else (shap_all[0] if shap_all else [])

    # 依赖图：TOP特征（importance 绝对值最大）+ 第二重要特征（交互着色），与 base64 依赖图
    # (shap.dependence_plot(top_idx, ..., interaction_index=importance_order[1])) 一致
    sorted_feats = sorted(importance.keys(), key=lambda k: abs(importance[k]), reverse=True)
    top_feature = sorted_feats[0] if sorted_feats else None
    interaction_feature = sorted_feats[1] if len(sorted_feats) > 1 else None
    top_idx = feature_names.index(top_feature) if top_feature in feature_names else None
    inter_idx = feature_names.index(interaction_feature) if interaction_feature in feature_names else None
    dep_data = []
    dep_shap = []
    dep_interaction = []
    if top_idx is not None:
        dep_data = [row[top_idx] for row in data_all if isinstance(row, list) and top_idx < len(row)]
        dep_shap = [row[top_idx] for row in shap_all if isinstance(row, list) and top_idx < len(row)]
        if inter_idx is not None:
            dep_interaction = [row[inter_idx] for row in data_all if isinstance(row, list) and inter_idx < len(row)]

    return {
        "SHAP_summary": {
            "feature_names": feature_names,
            "importance": importance,
            "samples": {
                "data": data_all,
                "shap_values": shap_all,
            },
        },
        "SHAP_waterfall": {
            "feature_names": feature_names,
            "base_value": base_value,
            "sample_id": sample_idx,
            "data": w_data,
            "shap_values": w_shap,
        },
        "SHAP_dependence": {
            "feature_names": feature_names,
            "top_feature": top_feature,
            "interaction_feature": interaction_feature,
            "data": dep_data,
            "shap_values": dep_shap,
            "interaction_data": dep_interaction,
        },
    }


def merge_chart_data_json(corr_json: str, pred_actual_json: str, shap_json: str, out_json: str) -> dict:
    """合并 corr_data.json / pred_actual_data.json / shap_data.json 为统一 chart_data.json，并删除原文件，
    同时将图表数据按性质合并到同目录的 output.json（顶层键用 conclusion_data，避免与 base64 的 conclusion 冲突）。

    增量保留：chart_data.json 已存在时，本轮未训练的性质图表数据与 conclusion_data 指标原样保留
    （与 output.json 的增量合并一致），仅更新/覆盖本轮选中的性质。

    每个性质节点包含与 figures.json（base64）对应的数据：
      - 簇状条形图_bar       : 训练集/测试集每个样品的 actual/pred（与 plot_clustered_bar_each_property 一致）
      - 训练测试集对比_scatter : 训练集/测试集样本的 actual/pred 散点数据 + metrics
      - 相关性分析           : 特征-性能相关系数
      - SHAP_summary         : 全局特征重要性（含所有样本 data/shap_values）
      - SHAP_waterfall       : 已挑选样本的瀑布数据
      - SHAP_dependence      : TOP特征依赖数据（含第二交互特征）
    顶层 conclusion_data 区分 r2/mape/pearson 三组（各含训练集+测试集）。
    """
    corr_data = _read_json_file(corr_json) or {}
    pred_actual_data = _read_json_file(pred_actual_json) or {}
    shap_data = _read_json_file(shap_json) or {}

    # 增量保留：读取已有 chart_data.json，未在本轮训练的性质数据原样保留（与 output.json 的增量合并一致）
    merged = {}
    _out_path = Path(out_json)
    if _out_path.exists():
        try:
            with open(_out_path, 'r', encoding='utf-8') as f:
                _existing_chart = json.load(f)
            if isinstance(_existing_chart, dict):
                merged = _existing_chart
        except (json.JSONDecodeError, OSError):
            merged = {}

    all_props = sorted(set(list(corr_data.keys()) + list(pred_actual_data.keys()) + list(shap_data.keys())))
    for prop in all_props:
        if prop == '生成时间':
            continue
        node = {}
        # 相关性分析（corr_data 每个性质下为 {feat: corr}）
        corr_entry = corr_data.get(prop)
        if isinstance(corr_entry, dict) and corr_entry:
            node["相关性分析"] = corr_entry
        # 训练测试集对比_scatter + 簇状条形图_bar（pred_actual_data）
        pa_entry = pred_actual_data.get(prop)
        if isinstance(pa_entry, dict):
            metrics_node = pa_entry.get('metrics')
            # 簇状条形图_bar：每个样品的实际值 vs 预测值柱状图（与 base64 图 plot_clustered_bar_each_property 一致）
            bar_node = {}
            for src in ('训练集', '测试集'):
                if isinstance(pa_entry.get(src), dict):
                    bar_node[src] = pa_entry[src]
            if bar_node:
                node["簇状条形图_bar"] = bar_node
            # 训练测试集对比_scatter：散点图数据（samples/actual/pred）+ 性能指标
            scatter_node = {}
            if isinstance(metrics_node, dict) and metrics_node:
                scatter_node["metrics"] = metrics_node
            for src in ('训练集', '测试集'):
                if isinstance(pa_entry.get(src), dict):
                    scatter_node[src] = pa_entry[src]
            if scatter_node:
                node["训练测试集对比_scatter"] = scatter_node
        # SHAP 三图
        shap_entry = shap_data.get(prop)
        if isinstance(shap_entry, dict):
            node.update(_build_shap_chart_data(shap_entry))
        if node:
            merged[prop] = node

    # conclusion_data：r2/mape/pearson 三组数据（训练集+测试集），命名避免与 base64 的 conclusion 键冲突；
    # 增量保留：本轮选中的性质逐键覆盖，未选中的性质指标保留
    new_conclusion = _build_conclusion_data(pred_actual_data)
    old_conclusion = merged.get("conclusion_data")
    if isinstance(old_conclusion, dict):
        for _group, _props in new_conclusion.items():
            _old_group = old_conclusion.get(_group)
            if isinstance(_old_group, dict):
                _old_group.update(_props)
            else:
                old_conclusion[_group] = _props
        merged["conclusion_data"] = old_conclusion
    else:
        merged["conclusion_data"] = new_conclusion

    out_json = Path(out_json)
    os.makedirs(out_json.parent, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"图表数据已统一合并至：{out_json}")

    # 按性质合并图表数据到 output.json（顶层 conclusion_data；各性质下追加图数据键）
    output_json = out_json.parent / "output.json"
    if output_json.exists():
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            output_data = {}
        if not isinstance(output_data, dict):
            output_data = {}
        for prop, node in merged.items():
            if prop == "conclusion_data":
                output_data[prop] = node
            elif isinstance(output_data.get(prop), dict):
                output_data[prop].update(node)
            else:
                output_data[prop] = node
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"图表数据已按性质合并到：{output_json}")
    else:
        print(f"警告：未找到{output_json}，图表数据未合并到output.json。")

    # 删除分散数据文件（不再单独输出）
    for p in (corr_json, pred_actual_json, shap_json):
        if p and os.path.exists(p):
            try:
                os.remove(p)
                print(f"已删除分散数据文件：{p}")
            except OSError as e:
                print(f"删除分散数据文件失败：{p} - {e}")
    return merged
