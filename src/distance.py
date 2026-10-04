"""Pairwise distance computation and violation logic."""
from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np
from scipy.spatial.distance import cdist


class DistanceAnalyzer:
    """Finds pairs of people standing closer than a given threshold."""

    def evaluate_violations(
        self, transformed_points: np.ndarray, threshold_px: float
    ) -> Tuple[Set[int], List[Tuple[int, int]]]:
        """Evaluate social distancing violations.

        Args:
            transformed_points: ``(N, 2)`` positions on the bird's-eye plane.
            threshold_px: Minimum safe distance in bird's-eye pixels.

        Returns:
            ``(violating_indices, violating_pairs)`` where pairs are ``(i, j)``
            with ``i < j``. Both are empty when fewer than two people exist.
        """
        pts = np.asarray(transformed_points, dtype=np.float32).reshape(-1, 2)
        n = pts.shape[0]
        if n < 2:
            return set(), []

        dist = cdist(pts, pts, metric="euclidean")
        i_idx, j_idx = np.triu_indices(n, k=1)  # each pair once, no self-pairs
        close = dist[i_idx, j_idx] < threshold_px

        i_bad, j_bad = i_idx[close], j_idx[close]
        pairs = list(zip(i_bad.tolist(), j_bad.tolist()))
        indices = set(i_bad.tolist()) | set(j_bad.tolist())
        return indices, pairs
