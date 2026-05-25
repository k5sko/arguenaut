"""Perspective-quality filtering for per-prompt discovery.

Two activation-space cleanups applied *before* PCA, plus a post-PCA guard:

  • dedup_and_trim   — drop near-duplicate perspectives and centroid outliers,
                       so redundant mass / off-topic points don't create or
                       dominate a principal component.
  • flag_outlier_components — after PCA, mark any component whose variance is
                       driven by one or two points, so a single outlier can't
                       masquerade as a discovered axis.

All operations are conservative (remove only clear cases) and report what they
removed so the effect is transparent rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FilterReport:
    n_in: int
    kept: np.ndarray          # bool mask over the input rows
    n_outliers: int
    n_duplicates: int

    @property
    def n_kept(self) -> int:
        return int(self.kept.sum())


def dedup_and_trim(
    X: np.ndarray,
    *,
    dup_cosine: float = 0.97,
    outlier_z: float = 3.5,
    min_keep: int = 6,
) -> FilterReport:
    """Return a keep-mask over rows of X ([N, d] activations at one layer).

    Centroid outliers (robust MAD z-score on distance-from-mean) and near-
    duplicates (cosine on mean-centred vectors — centring removes the base
    model's anisotropy so the similarity is meaningful) are dropped. Never drops
    below `min_keep` rows (PCA needs a cloud); if a cut would, it's relaxed.
    """
    n = X.shape[0]
    keep = np.ones(n, dtype=bool)
    if n <= min_keep:
        return FilterReport(n_in=n, kept=keep, n_outliers=0, n_duplicates=0)

    Xc = X - X.mean(axis=0, keepdims=True)

    # ── centroid outliers ───────────────────────────────────────────────────
    dist = np.linalg.norm(Xc, axis=1)
    med = np.median(dist)
    mad = np.median(np.abs(dist - med)) + 1e-9
    z = 0.6745 * (dist - med) / mad           # robust z-score
    outlier = z > outlier_z
    # don't let outlier removal drop us below min_keep
    if outlier.sum() and (n - outlier.sum()) >= min_keep:
        keep &= ~outlier
    else:
        outlier = np.zeros(n, dtype=bool)
    n_outliers = int(outlier.sum())

    # ── near-duplicates (greedy, among survivors) ────────────────────────────
    norm = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)
    selected: list[int] = []
    duplicate = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(keep):
        if any(float(norm[i] @ norm[j]) > dup_cosine for j in selected):
            duplicate[i] = True
        else:
            selected.append(int(i))
    # relax dedup if it would undercut min_keep
    if len(selected) >= min_keep:
        keep &= ~duplicate
    else:
        duplicate = np.zeros(n, dtype=bool)
    n_duplicates = int(duplicate.sum())

    return FilterReport(n_in=n, kept=keep, n_outliers=n_outliers, n_duplicates=n_duplicates)


def flag_outlier_components(
    scores: np.ndarray,            # [N, n_components] PCA projections
    *,
    dominance: float = 0.45,
) -> list[bool]:
    """For each component, True if a single point contributes more than
    `dominance` of that component's total squared projection — i.e. the "axis"
    is really one outlier, not a shared direction of variation."""
    flags: list[bool] = []
    for c in range(scores.shape[1]):
        s2 = scores[:, c] ** 2
        total = float(s2.sum()) + 1e-12
        flags.append(float(s2.max()) / total > dominance)
    return flags
