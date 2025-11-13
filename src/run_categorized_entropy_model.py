"""分类熵权法模型运行脚本

实现按分类分别计算熵权，然后合并各分类分数的科学评估方法。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None

from category_mapping import (
    CATEGORY_KEYWORDS,
    classify_indicators,
    get_category_indicators,
)
from data_cleaning import export_data, load_all_indicators
from entropy_model import EntropyWeightModel, guess_orientation

LOGGER = logging.getLogger(__name__)


def build_feature_matrix(
    long_frame: pd.DataFrame,
    year: int,
    min_feature_coverage: float,
    min_state_coverage: float,
) -> pd.DataFrame:
    """构建特征矩阵（州 × 指标）。"""
    year_frame = long_frame[long_frame["year"] == year]
    pivot = year_frame.pivot_table(
        index="region", columns="indicator_id", values="value", aggfunc="mean"
    )
    if "US" in pivot.index:
        pivot = pivot.drop(index="US")
        LOGGER.info("Excluded national aggregate 'US' from scoring matrix")
    feature_coverage = pivot.notna().mean(axis=0)
    pivot = pivot.loc[:, feature_coverage >= min_feature_coverage]
    if pivot.shape[1] == 0:
        raise RuntimeError(
            "No indicators satisfied the feature coverage threshold. "
            "Lower --min-feature-coverage or check data availability."
        )
    state_coverage = pivot.notna().mean(axis=1)
    pivot = pivot.loc[state_coverage >= min_state_coverage]
    if pivot.shape[0] == 0:
        raise RuntimeError(
            "No states satisfied the state coverage threshold. "
            "Lower --min-state-coverage or check data availability."
        )
    LOGGER.info(
        "Feature matrix: %d states x %d indicators",
        pivot.shape[0],
        pivot.shape[1],
    )
    return pivot.sort_index()


def ensure_dirs(base: Path) -> Dict[str, Path]:
    """确保输出目录存在。"""
    dirs = {
        "intermediate": base / "intermediate",
        "model": base / "model",
        "figures": base / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def compute_category_scores(
    features: pd.DataFrame,
    category: str,
    category_indicators: List[str],
    orientation_map: Dict[str, str],
    metadata: pd.DataFrame,
) -> tuple[pd.Series, EntropyWeightModel, pd.Series]:
    """计算单个分类的熵权分数。
    
    Args:
        features: 完整的特征矩阵
        category: 分类名称
        category_indicators: 该分类下的指标ID列表
        orientation_map: 指标方向映射
        metadata: 指标元数据
        
    Returns:
        (分类分数, 分类模型, 分类权重)
    """
    # 筛选该分类下的指标
    available_indicators = [
        ind for ind in category_indicators if ind in features.columns
    ]
    
    if len(available_indicators) == 0:
        LOGGER.warning("分类 '%s' 没有可用的指标", category)
        return None, None, None
    
    category_features = features[available_indicators].copy()
    
    # 移除全为NaN的列
    category_features = category_features.loc[:, category_features.notna().any(axis=0)]
    
    if category_features.shape[1] == 0:
        LOGGER.warning("分类 '%s' 没有有效数据", category)
        return None, None, None
    
    LOGGER.info(
        "分类 '%s': %d 个指标, %d 个州",
        category,
        category_features.shape[1],
        category_features.shape[0],
    )
    
    # 构建该分类的方向映射
    category_orientation = {
        col: orientation_map.get(col, "benefit")
        for col in category_features.columns
    }
    
    # 计算熵权
    category_model = EntropyWeightModel()
    category_scores = category_model.fit_transform(
        category_features, category_orientation
    )
    category_weights = category_model.weights_
    
    return category_scores, category_model, category_weights


def merge_category_scores(
    category_scores_dict: Dict[str, pd.Series],
) -> pd.Series:
    """合并各分类的分数。
    
    使用熵权法对各分类分数进行加权合并。
    
    Args:
        category_scores_dict: 字典，key为分类名称，value为该分类的分数Series
        
    Returns:
        合并后的综合分数
    """
    # 过滤掉None值
    valid_scores = {
        cat: scores
        for cat, scores in category_scores_dict.items()
        if scores is not None
    }
    
    if len(valid_scores) == 0:
        raise RuntimeError("没有有效的分类分数可以合并")
    
    # 将所有分类分数合并为DataFrame
    scores_df = pd.DataFrame(valid_scores)
    
    # 使用熵权法计算各分类的权重
    merge_model = EntropyWeightModel()
    # 所有分类分数都是benefit类型（分数越高越好）
    merge_orientation = {cat: "benefit" for cat in scores_df.columns}
    merge_weights = merge_model.fit(scores_df, merge_orientation)
    
    # 计算加权综合分数
    normalized_scores = scores_df.mul(merge_weights, axis=1)
    composite_scores = normalized_scores.sum(axis=1)
    
    # 归一化到0-100
    max_score = composite_scores.max()
    if max_score > 0:
        composite_scores = composite_scores / max_score * 100
    
    return composite_scores.sort_values(ascending=False), merge_weights


# Category name mapping to English
CATEGORY_EN_NAMES = {
    "能源消费类": "Energy Consumption",
    "能源价格类": "Energy Price & Expenditure",
    "碳排放类": "Carbon Emissions",
    "经济人口类": "Economy & Population",
    "气候环境类": "Climate & Environment",
    "城市水资源类": "Urban Water Resources",
    "航空出行类": "Air Travel Connectivity",
    "废弃物管理类": "Waste Management",
}


def plot_category_weights(
    category_weights: pd.Series, out_path: Path
) -> None:
    """绘制分类权重图。"""
    # Convert category names to English
    category_weights_en = category_weights.copy()
    category_weights_en.index = [
        CATEGORY_EN_NAMES.get(cat, cat) for cat in category_weights_en.index
    ]
    
    plt.figure(figsize=(10, 6))
    if sns:
        sns.barplot(
            x=category_weights_en.values,
            y=category_weights_en.index,
            color="#4c72b0",
        )
    else:
        plt.barh(category_weights_en.index, category_weights_en.values)
    plt.xlabel("Weight")
    plt.ylabel("Category")
    plt.title("Category Weights (Entropy Weight Method)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_category_comparison(
    category_scores_dict: Dict[str, pd.Series],
    top_k: int,
    out_path: Path,
) -> None:
    """绘制各分类分数对比图（前k个州）。"""
    # 获取前k个州
    all_states = set()
    for scores in category_scores_dict.values():
        if scores is not None:
            all_states.update(scores.head(top_k).index)
    
    # 构建对比数据
    comparison_data = []
    for state in sorted(all_states)[:top_k]:
        for category, scores in category_scores_dict.items():
            if scores is not None and state in scores.index:
                comparison_data.append(
                    {
                        "State": state,
                        "Category": CATEGORY_EN_NAMES.get(category, category),
                        "Score": scores[state],
                    }
                )
    
    if not comparison_data:
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 按综合分数排序
    if len(comparison_df) > 0:
        plt.figure(figsize=(12, 8))
        if sns:
            sns.barplot(
                data=comparison_df,
                x="State",
                y="Score",
                hue="Category",
            )
        else:
            # 手动绘制
            states = comparison_df["State"].unique()
            categories = comparison_df["Category"].unique()
            x = range(len(states))
            width = 0.8 / len(categories)
            for i, cat in enumerate(categories):
                cat_data = comparison_df[comparison_df["Category"] == cat]
                values = [cat_data[cat_data["State"] == s]["Score"].values[0] if len(cat_data[cat_data["State"] == s]) > 0 else 0 for s in states]
                plt.bar(
                    [xi + i * width - width * (len(categories) - 1) / 2 for xi in x],
                    values,
                    width,
                    label=cat,
                )
            plt.xticks(x, states, rotation=45)
        plt.xlabel("State")
        plt.ylabel("Score")
        plt.title(f"Top {top_k} States: Category Score Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()


def plot_category_score_distribution(
    category_scores_dict: Dict[str, pd.Series],
    out_path: Path,
) -> None:
    """绘制各分类分数的分布图。"""
    valid_scores = {
        CATEGORY_EN_NAMES.get(cat, cat): scores
        for cat, scores in category_scores_dict.items()
        if scores is not None
    }
    
    if not valid_scores:
        return
    
    fig, axes = plt.subplots(
        len(valid_scores), 1, figsize=(10, 3 * len(valid_scores))
    )
    if len(valid_scores) == 1:
        axes = [axes]
    
    for idx, (category_name, scores) in enumerate(valid_scores.items()):
        if sns:
            sns.histplot(scores, bins=15, kde=True, ax=axes[idx], color="#377eb8")
        else:
            axes[idx].hist(scores, bins=15, color="#377eb8", alpha=0.8)
        axes[idx].set_xlabel("Score")
        axes[idx].set_ylabel("Number of States")
        axes[idx].set_title(f"{category_name} Score Distribution")
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_composite_scores(
    composite_scores: pd.Series,
    out_path: Path,
    top_k: int = 15,
) -> None:
    """绘制综合分数排名图。"""
    top = composite_scores.sort_values(ascending=False).head(top_k).iloc[::-1]
    plt.figure(figsize=(8, 6))
    if sns:
        sns.barplot(x=top.values, y=top.index, color="#009688")
    else:
        plt.barh(top.index, top.values)
    plt.xlabel("Composite Score")
    plt.ylabel("State")
    plt.title(f"Top {top_k} States: Composite Scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_category_indicator_weights(
    category_weights_dict: Dict[str, pd.Series],
    metadata: pd.DataFrame,
    out_path: Path,
    top_k: int = 10,
) -> None:
    """绘制各分类内指标权重图（每个分类的top指标）。"""
    valid_categories = {
        cat: weights
        for cat, weights in category_weights_dict.items()
        if weights is not None and len(weights) > 0
    }
    
    if not valid_categories:
        return
    
    n_categories = len(valid_categories)
    fig, axes = plt.subplots(
        n_categories, 1, figsize=(12, 4 * n_categories)
    )
    if n_categories == 1:
        axes = [axes]
    
    for idx, (category, weights) in enumerate(valid_categories.items()):
        category_en = CATEGORY_EN_NAMES.get(category, category)
        
        # 合并元数据
        merged = (
            weights.rename("weight")
            .reset_index()
            .rename(columns={"index": "indicator_id"})
            .merge(metadata, how="left", on="indicator_id")
        )
        merged["indicator_label"] = merged["indicator_name"].fillna(
            merged["indicator_id"]
        )
        top = merged.nlargest(top_k, "weight")
        
        if sns:
            sns.barplot(
                data=top, y="indicator_label", x="weight", ax=axes[idx], color="#4c72b0"
            )
        else:
            axes[idx].barh(top["indicator_label"], top["weight"])
        axes[idx].set_xlabel("Weight")
        axes[idx].set_ylabel("")
        axes[idx].set_title(f"Top {top_k} Indicators: {category_en}")
        axes[idx].grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_category_correlation(
    category_scores_dict: Dict[str, pd.Series],
    composite_scores: pd.Series,
    out_path: Path,
) -> None:
    """绘制各分类分数与综合分数的相关性散点图。"""
    valid_scores = {
        CATEGORY_EN_NAMES.get(cat, cat): scores
        for cat, scores in category_scores_dict.items()
        if scores is not None
    }
    
    if not valid_scores:
        return
    
    n_categories = len(valid_scores)
    cols = 2
    rows = (n_categories + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten() if n_categories > 1 else [axes]
    
    for idx, (category_name, scores) in enumerate(valid_scores.items()):
        # 对齐索引
        common_states = scores.index.intersection(composite_scores.index)
        cat_scores = scores[common_states]
        comp_scores = composite_scores[common_states]
        
        # 计算相关系数
        corr = cat_scores.corr(comp_scores)
        
        axes[idx].scatter(cat_scores, comp_scores, alpha=0.6, color="#377eb8")
        axes[idx].set_xlabel(f"{category_name} Score")
        axes[idx].set_ylabel("Composite Score")
        axes[idx].set_title(f"{category_name} vs Composite (r={corr:.3f})")
        axes[idx].grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(cat_scores, comp_scores, 1)
        p = np.poly1d(z)
        axes[idx].plot(
            cat_scores.sort_values(),
            p(cat_scores.sort_values()),
            "r--",
            alpha=0.8,
            linewidth=2,
        )
    
    # 隐藏多余的子图
    for idx in range(n_categories, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_category_heatmap(
    category_scores_dict: Dict[str, pd.Series],
    top_k: int,
    out_path: Path,
) -> None:
    """绘制各州在不同分类中的排名热力图。"""
    valid_scores = {
        CATEGORY_EN_NAMES.get(cat, cat): scores
        for cat, scores in category_scores_dict.items()
        if scores is not None
    }
    
    if not valid_scores:
        return
    
    # 获取所有州的排名
    all_states = set()
    for scores in valid_scores.values():
        all_states.update(scores.head(top_k).index)
    
    # 构建排名矩阵
    rank_data = []
    for state in sorted(all_states)[:top_k]:
        row = {"State": state}
        for category_name, scores in valid_scores.items():
            if state in scores.index:
                # 计算排名（分数越高排名越靠前）
                rank = (scores >= scores[state]).sum()
                row[category_name] = rank
            else:
                row[category_name] = np.nan
        rank_data.append(row)
    
    rank_df = pd.DataFrame(rank_data).set_index("State")
    
    plt.figure(figsize=(max(10, len(valid_scores) * 2), max(6, top_k * 0.5)))
    if sns:
        sns.heatmap(
            rank_df.T,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            cbar_kws={"label": "Rank (lower is better)"},
            linewidths=0.5,
        )
    else:
        # 简单的热力图
        plt.imshow(rank_df.T.values, cmap="YlOrRd", aspect="auto")
        plt.colorbar(label="Rank (lower is better)")
        plt.xticks(range(len(rank_df.index)), rank_df.index, rotation=45)
        plt.yticks(range(len(rank_df.columns)), rank_df.columns)
    
    plt.xlabel("State")
    plt.ylabel("Category")
    plt.title(f"Top {top_k} States: Category Rankings Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_category_boxplot(
    category_scores_dict: Dict[str, pd.Series],
    out_path: Path,
) -> None:
    """绘制各分类分数的箱线图对比。"""
    valid_scores = {
        CATEGORY_EN_NAMES.get(cat, cat): scores
        for cat, scores in category_scores_dict.items()
        if scores is not None
    }
    
    if not valid_scores:
        return
    
    # 准备数据
    plot_data = []
    for category_name, scores in valid_scores.items():
        for state, score in scores.items():
            plot_data.append({"Category": category_name, "Score": score})
    
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 6))
    if sns:
        sns.boxplot(data=plot_df, x="Category", y="Score", palette="Set2")
        sns.stripplot(
            data=plot_df, x="Category", y="Score", color="black", alpha=0.3, size=3
        )
    else:
        categories = plot_df["Category"].unique()
        data_list = [plot_df[plot_df["Category"] == cat]["Score"].values for cat in categories]
        plt.boxplot(data_list, labels=categories)
    plt.xlabel("Category")
    plt.ylabel("Score")
    plt.title("Category Score Distribution Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def serialize_category_results(
    category_models: Dict[str, EntropyWeightModel],
    category_weights: Dict[str, pd.Series],
    merge_weights: pd.Series,
    output_dir: Path,
    year: int,
) -> None:
    """序列化分类结果。"""
    # 保存各分类的权重
    category_weights_df = pd.DataFrame(category_weights).T
    category_weights_df.to_csv(
        output_dir / f"category_indicator_weights_{year}.csv"
    )
    
    # 保存分类合并权重
    merge_weights.to_csv(
        output_dir / f"category_merge_weights_{year}.csv", header=["weight"]
    )
    
    # 保存各分类模型
    category_models_dict = {}
    for category, model in category_models.items():
        if model is not None:
            category_models_dict[category] = {
                "weights": model.weights_.to_dict(),
                "feature_stats": {
                    feature: {
                        "minimum": stat.minimum,
                        "maximum": stat.maximum,
                        "orientation": stat.orientation,
                        "weight": stat.weight,
                    }
                    for feature, stat in model.feature_stats_.items()
                },
            }
    
    (output_dir / f"category_models_{year}.json").write_text(
        json.dumps(category_models_dict, indent=2, ensure_ascii=False)
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="构建分类熵权法评分模型。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="包含原始Excel文件的目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="输出目录。",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="构建特征矩阵的年份。",
    )
    parser.add_argument(
        "--min-feature-coverage",
        type=float,
        default=0.85,
        help="保留特征所需的最小州数据覆盖率。",
    )
    parser.add_argument(
        "--min-state-coverage",
        type=float,
        default=0.8,
        help="对州进行评分所需的最小特征覆盖率。",
    )
    return parser.parse_args()


def main() -> None:
    """主函数。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    dirs = ensure_dirs(args.output_dir)

    LOGGER.info("从 %s 加载原始工作簿", args.data_dir)
    long_frame, metadata = load_all_indicators(args.data_dir)
    export_data(long_frame, metadata, dirs["intermediate"])

    # 构建特征矩阵
    features = build_feature_matrix(
        long_frame, args.year, args.min_feature_coverage, args.min_state_coverage
    )
    features.to_csv(dirs["model"] / f"feature_matrix_{args.year}.csv")

    # 构建指标分类映射
    indicator_names = dict(
        zip(metadata["indicator_id"], metadata["indicator_name"])
    )
    classification = classify_indicators(indicator_names, metadata=metadata)
    
    # 统计分类结果
    classification_counts = {}
    for cat in CATEGORY_KEYWORDS.keys():
        indicators = get_category_indicators(classification, cat)
        available = [ind for ind in indicators if ind in features.columns]
        classification_counts[cat] = {
            "total": len(indicators),
            "available": len(available),
        }
        LOGGER.info(
            "分类 '%s': 总共 %d 个指标, 可用 %d 个指标",
            cat,
            len(indicators),
            len(available),
        )
    
    # 保存分类映射
    classification_df = pd.DataFrame(
        [
            {"indicator_id": ind_id, "category": cat}
            for ind_id, cat in classification.items()
        ]
    )
    classification_df.to_csv(
        dirs["intermediate"] / f"indicator_classification_{args.year}.csv",
        index=False,
    )

    # 构建方向映射
    orientation_map = {
        row.indicator_id: guess_orientation(row.indicator_name)
        for row in metadata.itertuples()
    }

    # 对每个分类分别计算熵权分数
    category_scores_dict = {}
    category_models = {}
    category_weights_dict = {}
    
    for category in CATEGORY_KEYWORDS.keys():
        category_indicators = get_category_indicators(classification, category)
        scores, model, weights = compute_category_scores(
            features,
            category,
            category_indicators,
            orientation_map,
            metadata,
        )
        category_scores_dict[category] = scores
        category_models[category] = model
        category_weights_dict[category] = weights
        
        if scores is not None:
            scores.to_csv(
                dirs["model"] / f"state_scores_{category}_{args.year}.csv",
                header=["score"],
            )

    # 合并各分类分数
    composite_scores, merge_weights = merge_category_scores(category_scores_dict)
    composite_scores.to_csv(
        dirs["model"] / f"state_scores_categorized_{args.year}.csv",
        header=["score"],
    )
    
    LOGGER.info("合并后的分类权重:")
    for cat, weight in merge_weights.items():
        LOGGER.info("  %s: %.4f", cat, weight)

    # 保存分类结果
    serialize_category_results(
        category_models, category_weights_dict, merge_weights, dirs["model"], args.year
    )

    # 绘制图表
    plot_category_weights(
        merge_weights, dirs["figures"] / f"category_weights_{args.year}.png"
    )
    plot_category_comparison(
        category_scores_dict,
        15,
        dirs["figures"] / f"category_comparison_{args.year}.png",
    )
    plot_category_score_distribution(
        category_scores_dict,
        dirs["figures"] / f"category_score_distribution_{args.year}.png",
    )
    plot_composite_scores(
        composite_scores,
        dirs["figures"] / f"composite_scores_top_{args.year}.png",
    )
    plot_category_indicator_weights(
        category_weights_dict,
        metadata,
        dirs["figures"] / f"category_indicator_weights_{args.year}.png",
    )
    plot_category_correlation(
        category_scores_dict,
        composite_scores,
        dirs["figures"] / f"category_correlation_{args.year}.png",
    )
    plot_category_heatmap(
        category_scores_dict,
        15,
        dirs["figures"] / f"category_heatmap_{args.year}.png",
    )
    plot_category_boxplot(
        category_scores_dict,
        dirs["figures"] / f"category_boxplot_{args.year}.png",
    )

    # 保存摘要
    summary = {
        "year": args.year,
        "states": composite_scores.shape[0],
        "categories": len([s for s in category_scores_dict.values() if s is not None]),
        "category_counts": classification_counts,
        "category_merge_weights": merge_weights.to_dict(),
        "top_state": composite_scores.index[0],
        "top_score": float(composite_scores.iloc[0]),
    }
    (dirs["model"] / f"categorized_summary_{args.year}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    LOGGER.info("完成！综合分数已保存到 state_scores_categorized_%d.csv", args.year)


if __name__ == "__main__":
    main()
