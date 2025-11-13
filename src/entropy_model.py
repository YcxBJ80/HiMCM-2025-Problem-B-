from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def guess_orientation(name: str) -> str:
    """Heuristic orientation guess: benefit (larger better) vs cost."""

    if not isinstance(name, str):
        return "benefit"
    lowered = name.lower()
    positive = [
        "renewable",
        "gdp",
        "production",
        "biofuel",
        "efficiency",
        "capacity",
        "population",
        "income",
    ]
    negative = [
        "emission",
        "co2",
        "intensity",
        "price",
        "expenditure",
        "consumption",
        "spending",
        "cost",
        "withdrawal",
        "waste",
    ]
    if any(token in lowered for token in positive):
        return "benefit"
    if any(token in lowered for token in negative):
        return "cost"
    return "benefit"


@dataclass
class FeatureStats:
    minimum: float
    maximum: float
    orientation: str
    weight: Optional[float] = None
    entropy: Optional[float] = None


@dataclass
class EntropyWeightModel:
    epsilon: float = 1e-12
    feature_stats_: Dict[str, FeatureStats] = field(default_factory=dict)
    weights_: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    entropy_: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    divergence_: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fitted_: bool = False

    def fit(self, frame: pd.DataFrame, orientation: Optional[Dict[str, str]] = None) -> pd.Series:
        """Fit weights using the provided feature matrix."""

        norm_frame, stats = self._normalize(frame, orientation)
        prob = self._probability(norm_frame)
        entropy = self._entropy(prob)
        divergence = 1 - entropy
        divergence = divergence.clip(lower=0)
        if divergence.sum() == 0:
            weights = pd.Series(1 / len(divergence), index=divergence.index)
        else:
            weights = divergence / divergence.sum()
        for feature, stat in stats.items():
            stat.weight = float(weights.get(feature, 0.0))
            stat.entropy = float(entropy.get(feature, 0.0))
        self.feature_stats_ = stats
        self.weights_ = weights
        self.entropy_ = entropy
        self.divergence_ = divergence
        self.fitted_ = True
        return weights

    def fit_transform(
        self, frame: pd.DataFrame, orientation: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        weights = self.fit(frame, orientation)
        norm_frame, _ = self._normalize(frame, orientation, reuse_stats=True)
        raw_scores = norm_frame.mul(weights, axis=1).sum(axis=1)
        max_score = raw_scores.max()
        if max_score > 0:
            scaled = raw_scores / max_score * 100
        else:
            scaled = raw_scores
        return scaled.sort_values(ascending=False)

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling transform.")
        norm_frame, _ = self._normalize(frame, reuse_stats=True)
        raw_scores = norm_frame.mul(self.weights_, axis=1).sum(axis=1)
        max_score = raw_scores.max()
        if max_score > 0:
            return raw_scores / max_score * 100
        return raw_scores

    def _normalize(
        self,
        frame: pd.DataFrame,
        orientation: Optional[Dict[str, str]] = None,
        reuse_stats: bool = False,
    ) -> tuple[pd.DataFrame, Dict[str, FeatureStats]]:
        stats: Dict[str, FeatureStats] = {}
        norm = pd.DataFrame(index=frame.index)
        if reuse_stats and self.feature_stats_:
            stats = self.feature_stats_
        for column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            if reuse_stats and column in stats:
                min_val = stats[column].minimum
                max_val = stats[column].maximum
                orient = stats[column].orientation
            else:
                min_val = float(series.min())
                max_val = float(series.max())
                orient = (orientation or {}).get(column, "benefit")
                stats[column] = FeatureStats(minimum=min_val, maximum=max_val, orientation=orient)
            if math.isnan(min_val) or math.isnan(max_val):
                norm[column] = 0.0
                continue
            if math.isclose(max_val, min_val):
                norm[column] = 1.0
                continue
            if orient == "cost":
                numerator = max_val - series
            else:
                numerator = series - min_val
            denom = max_val - min_val
            norm[column] = (numerator / denom).clip(lower=0)
        norm = norm.replace([np.inf, -np.inf], np.nan).fillna(0)
        return norm, stats

    def _probability(self, norm: pd.DataFrame) -> pd.DataFrame:
        col_sums = norm.sum(axis=0) + self.epsilon
        prob = norm.div(col_sums, axis=1).clip(lower=self.epsilon)
        return prob

    def _entropy(self, prob: pd.DataFrame) -> pd.Series:
        n_samples = prob.shape[0]
        k = 1.0 / math.log(n_samples)
        return -(k * (prob * np.log(prob)).sum(axis=0))

    def export(self, path: Path) -> None:
        payload = {
            "weights": self.weights_.to_dict(),
            "feature_stats": {
                feature: {
                    "minimum": stat.minimum,
                    "maximum": stat.maximum,
                    "orientation": stat.orientation,
                    "weight": stat.weight,
                }
                for feature, stat in self.feature_stats_.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
