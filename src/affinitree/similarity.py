from __future__ import annotations

"""
Similarity and distance utilities for the Affinitree project.

This module is responsible for constructing the pairwise distance matrix
used by the embedding and clustering steps.

Responsibilities:
- Take a normalized trait matrix (rows = individuals, cols = traits)
- Optionally apply per-trait distance weights from the configuration
- Compute a pairwise distance matrix using scikit-learn
- Provide simple helper functions for nearest-neighbor lookups

Main entry point:
- build_distance_matrix(df_norm, trait_columns, cfg)
"""

from typing import List, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from .config import AffinitreeConfig, TraitMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight handling
# ---------------------------------------------------------------------------


def _extract_trait_code_from_column(col_name: str) -> str:
    """
    Infer a trait code (e.g. 'NS') from a column name.

    By default this assumes columns are named like:
        'NS Total', 'HA Total', etc.

    The implementation takes the first token before whitespace. If your data
    uses a different naming convention, adjust this helper accordingly or
    provide an explicit mapping.
    """
    return col_name.split()[0]


def _build_weight_vector(
    trait_columns: List[str],
    cfg: AffinitreeConfig,
) -> np.ndarray:
    """
    Build a weight vector aligned with trait_columns using configuration.

    We look up each column's trait code in the configuration's TraitMeta
    entries and return their distance_weight values. If a column's code
    is not found in the config, a default weight of 1.0 is used and a
    warning is logged.

    Returns
    -------
    np.ndarray
        1D array of length len(trait_columns) with distance weights.
    """
    meta_by_code = cfg.trait_meta_by_code()
    weights: List[float] = []

    for col in trait_columns:
        code = _extract_trait_code_from_column(col)
        meta: TraitMeta | None = meta_by_code.get(code)
        if meta is None:
            logger.warning(
                "No TraitMeta found for column '%s' (code '%s'); using weight 1.0",
                col,
                code,
            )
            weights.append(1.0)
        else:
            weights.append(meta.distance_weight)

    weights_arr = np.asarray(weights, dtype=float)

    logger.debug(
        "Using distance weights for traits: %s",
        ", ".join(f"{c}={w:.3f}" for c, w in zip(trait_columns, weights_arr)),
    )

    return weights_arr


def _apply_weights_to_matrix(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Apply trait weights to a normalized trait matrix.

    For Euclidean-style distances, a common pattern is to scale each feature
    by sqrt(weight). The squared Euclidean distance in the scaled space then
    corresponds to a weighted Euclidean distance in the original space.

    Parameters
    ----------
    values
        2D array of shape (n_samples, n_traits) with normalized trait values.
    weights
        1D array of length n_traits with non-negative weights.

    Returns
    -------
    2D array
        Weighted matrix of the same shape as `values`.
    """
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array")

    if values.shape[1] != len(weights):
        raise ValueError(
            f"values has {values.shape[1]} columns but weights has {len(weights)} entries"
        )

    # Avoid negative weights; treat any negative as zero.
    weights_clipped = np.clip(weights, a_min=0.0, a_max=None)
    scale = np.sqrt(weights_clipped)

    # values shape: (n_samples, n_traits)
    # scale shape: (n_traits,)
    weighted = values * scale

    logger.info("Applied distance weights to trait matrix")

    return weighted


# ---------------------------------------------------------------------------
# Distance matrix construction
# ---------------------------------------------------------------------------


def build_distance_matrix(
    df_norm: pd.DataFrame,
    trait_columns: List[str],
    cfg: AffinitreeConfig,
) -> np.ndarray:
    """
    Build the pairwise distance matrix over the normalized trait matrix.

    Steps:
    1. Extract the numeric matrix from df_norm using trait_columns.
    2. Construct a per-trait weight vector from cfg.
    3. Apply the weights to the matrix.
    4. Use scikit-learn's pairwise_distances to compute the distance matrix.

    Parameters
    ----------
    df_norm
        DataFrame of normalized traits (rows = individuals, cols = traits).
        This should typically be the output of preprocessing.build_trait_matrix.
    trait_columns
        Ordered list of trait column names that define the feature space.
    cfg
        AffinitreeConfig specifying distance_metric and trait weights.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_samples, n_samples) where entry (i, j) is
        the distance between individual i and j.
    """
    if not trait_columns:
        raise ValueError("trait_columns must not be empty")

    # Extract raw numeric values
    X = df_norm[trait_columns].to_numpy(dtype=float)
    n_samples, n_features = X.shape

    if n_samples == 0:
        raise ValueError("No rows in normalized trait matrix; nothing to compare")

    logger.info(
        "Building distance matrix for %d individuals with %d traits",
        n_samples,
        n_features,
    )

    # Build and apply weights
    weights = _build_weight_vector(trait_columns, cfg)
    X_weighted = _apply_weights_to_matrix(X, weights)

    # Compute pairwise distances
    metric = cfg.distance_metric or "euclidean"
    logger.info("Computing pairwise distances with metric='%s'", metric)

    dist_matrix = pairwise_distances(X_weighted, metric=metric)

    logger.info(
        "Distance matrix computed: shape (%d, %d)", dist_matrix.shape[0], dist_matrix.shape[1]
    )

    return dist_matrix


# ---------------------------------------------------------------------------
# Nearest neighbors helpers
# ---------------------------------------------------------------------------


def nearest_neighbors(
    dist_matrix: np.ndarray,
    k: int = 5,
    include_self: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute indices and distances of k nearest neighbors for each row.

    Parameters
    ----------
    dist_matrix
        2D array of shape (n_samples, n_samples) with pairwise distances.
        Diagonal entries are assumed to be zero (distance to self).
    k
        Number of neighbors to return per row. If include_self is False,
        the k nearest *other* points are returned.
    include_self
        If True, self is allowed in the neighbor list. If False, self is
        excluded by ignoring the zero diagonal.

    Returns
    -------
    (indices, distances)
        - indices: int array of shape (n_samples, k)
        - distances: float array of shape (n_samples, k)

        indices[i, :] gives the row indices of i's k nearest neighbors.
    """
    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be square")

    n = dist_matrix.shape[0]
    if n == 0:
        raise ValueError("Empty distance matrix")

    if k <= 0:
        raise ValueError("k must be positive")

    # For each row, sort by distance
    # argsort along axis 1
    order = np.argsort(dist_matrix, axis=1)

    if not include_self:
        # The closest neighbor to each row is itself (distance 0), so we
        # skip the first column in the sorted order.
        order = order[:, 1 : k + 1]
    else:
        order = order[:, :k]

    # Gather distances corresponding to those indices
    row_indices = np.arange(n)[:, None]
    nn_distances = dist_matrix[row_indices, order]

    logger.debug(
        "Computed %d nearest neighbors per individual (include_self=%s)",
        k,
        include_self,
    )

    return order, nn_distances