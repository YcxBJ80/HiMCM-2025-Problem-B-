from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

from data_cleaning import export_data, load_all_indicators
from entropy_model import EntropyWeightModel, guess_orientation

LOGGER = logging.getLogger(__name__)


def build_feature_matrix(
    long_frame: pd.DataFrame,
    year: int,
    min_feature_coverage: float,
    min_state_coverage: float,
) -> pd.DataFrame:
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
    dirs = {
        "intermediate": base / "intermediate",
        "model": base / "model",
        "figures": base / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def plot_weights(weights: pd.Series, metadata: pd.DataFrame, out_path: Path, top_k: int = 20) -> None:
    merged = (
        weights.rename("weight")
        .reset_index()
        .rename(columns={"index": "indicator_id"})
        .merge(metadata, how="left", on="indicator_id")
    )
    merged["indicator_label"] = merged["indicator_name"].fillna(merged["indicator_id"])
    top = merged.nlargest(top_k, "weight")
    plt.figure(figsize=(10, 6))
    if sns:
        sns.barplot(data=top, y="indicator_label", x="weight", color="#4c72b0")
    else:  # pragma: no cover - fallback
        plt.barh(top["indicator_label"], top["weight"])
    plt.title("Top indicator weights")
    plt.xlabel("Weight")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_score_distribution(scores: pd.Series, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    if sns:
        sns.histplot(scores, bins=15, kde=True, color="#377eb8")
    else:  # pragma: no cover
        plt.hist(scores, bins=15, color="#377eb8", alpha=0.8)
    plt.xlabel("Composite score")
    plt.ylabel("Number of states")
    plt.title("Score distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_top_states(scores: pd.Series, out_path: Path, top_k: int = 15) -> None:
    top = scores.sort_values(ascending=False).head(top_k).iloc[::-1]
    plt.figure(figsize=(8, 6))
    if sns:
        sns.barplot(x=top.values, y=top.index, color="#009688")
    else:  # pragma: no cover
        plt.barh(top.index, top.values)
    plt.xlabel("Score")
    plt.ylabel("State")
    plt.title(f"Top {top_k} scoring states")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_entropy(entropy: pd.Series, metadata: pd.DataFrame, out_path: Path) -> None:
    merged = (
        entropy.rename("entropy")
        .reset_index()
        .rename(columns={"index": "indicator_id"})
        .merge(metadata, how="left", on="indicator_id")
    )
    merged["indicator_label"] = merged["indicator_name"].fillna(merged["indicator_id"])
    merged = merged.sort_values("entropy", ascending=False)
    height = max(6, 0.25 * len(merged))
    plt.figure(figsize=(10, height))
    if sns:
        sns.barplot(
            data=merged,
            y="indicator_label",
            x="entropy",
            color="#ff7f0e",
        )
    else:  # pragma: no cover
        plt.barh(merged["indicator_label"], merged["entropy"])
    plt.xlabel("Information entropy")
    plt.ylabel("")
    plt.title("Indicator information entropy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def serialize_feature_stats(stats: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for indicator_id, payload in stats.items():
        minimum = payload["minimum"] if isinstance(payload, dict) else payload.minimum
        maximum = payload["maximum"] if isinstance(payload, dict) else payload.maximum
        orientation = (
            payload["orientation"] if isinstance(payload, dict) else payload.orientation
        )
        weight = payload.get("weight") if isinstance(payload, dict) else payload.weight
        entropy = payload.get("entropy") if isinstance(payload, dict) else payload.entropy
        rows.append(
            {
                "indicator_id": indicator_id,
                "min_value": minimum,
                "max_value": maximum,
                "orientation": orientation,
                "weight": weight or 0,
                "entropy": entropy or 0,
            }
        )
    return pd.DataFrame(rows).sort_values("weight", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build entropy-weight scoring model.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory with raw Excel files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Base output directory.")
    parser.add_argument("--year", type=int, default=2023, help="Year to build the feature matrix on.")
    parser.add_argument(
        "--min-feature-coverage",
        type=float,
        default=0.85,
        help="Minimum share of states with valid data required to keep a feature.",
    )
    parser.add_argument(
        "--min-state-coverage",
        type=float,
        default=0.8,
        help="Minimum share of features required for a state to be scored.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    dirs = ensure_dirs(args.output_dir)

    LOGGER.info("Loading raw workbooks from %s", args.data_dir)
    long_frame, metadata = load_all_indicators(args.data_dir)
    export_data(long_frame, metadata, dirs["intermediate"])

    features = build_feature_matrix(
        long_frame, args.year, args.min_feature_coverage, args.min_state_coverage
    )
    features.to_csv(dirs["model"] / f"feature_matrix_{args.year}.csv")

    orientation_map = {
        row.indicator_id: guess_orientation(row.indicator_name)
        for row in metadata.itertuples()
    }
    orientation_subset = {col: orientation_map.get(col, "benefit") for col in features.columns}

    model = EntropyWeightModel()
    scores = model.fit_transform(features, orientation_subset)
    weights = model.weights_
    weights.to_csv(dirs["model"] / f"indicator_weights_{args.year}.csv", header=["weight"])
    scores.to_csv(dirs["model"] / f"state_scores_{args.year}.csv", header=["score"])
    model.export(dirs["model"] / f"entropy_model_{args.year}.json")

    stats_df = serialize_feature_stats(model.feature_stats_)
    stats_df.to_csv(dirs["model"] / f"feature_stats_{args.year}.csv", index=False)

    plot_weights(
        weights,
        metadata,
        dirs["figures"] / f"indicator_weights_top_{args.year}.png",
    )
    plot_entropy(
        model.entropy_,
        metadata,
        dirs["figures"] / f"indicator_entropy_{args.year}.png",
    )
    plot_score_distribution(
        scores, dirs["figures"] / f"score_distribution_{args.year}.png"
    )
    plot_top_states(scores, dirs["figures"] / f"top_states_{args.year}.png")

    summary = {
        "year": args.year,
        "states": scores.shape[0],
        "features": weights.shape[0],
        "top_state": scores.index[0],
        "top_score": float(scores.iloc[0]),
    }
    (dirs["model"] / f"summary_{args.year}.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
