"""Quantitative separation / validation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def silhouette_pro_con(points_2d: np.ndarray, labels: np.ndarray) -> float:
    """Standard silhouette score where labels are pro/con (-1 vs +1)."""
    uniq = set(labels.tolist())
    if uniq != {-1, 1} and uniq != {0, 1} and len(uniq) != 2:
        return float("nan")
    if len(points_2d) < 4:
        return float("nan")
    return float(silhouette_score(points_2d, labels))


def correlation_score(predicted_rank: np.ndarray, actual_score: np.ndarray) -> float:
    """Spearman correlation between predicted ranking and actual projected scores.

    Used by the labeling verification loop: predicted_rank is [+1, +1, +1, +1, -1, -1, -1, -1]
    or any vector aligned with predicted polarity; actual_score is the PCA projection.
    """
    from scipy.stats import spearmanr

    if len(predicted_rank) != len(actual_score):
        raise ValueError("length mismatch")
    rho, _ = spearmanr(predicted_rank, actual_score)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def stance_to_pro_con(stance: str) -> int:
    """Map stance strings to -1/+1 for silhouette scoring; 0 means ignore."""
    table = {
        "strong_agree": +1,
        "qualified_agree": +1,
        "neutral_uncertain": 0,
        "qualified_disagree": -1,
        "strong_disagree": -1,
        "orthogonal_reframe": 0,
    }
    return table.get(stance, 0)
