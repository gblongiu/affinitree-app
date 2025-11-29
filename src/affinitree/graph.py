from __future__ import annotations

"""
Embedding utilities for the Affinitree project.

This module is responsible for projecting individuals from trait space into
two dimensional coordinates that can be used in the visualization.

Responsibilities:
- Take a pairwise distance matrix between individuals
- Use a configurable embedding method to obtain 2D coordinates
- Return a small DataFrame with x and y columns aligned to the input index

Currently supported method:
- "mds"  Multidimensional scaling from scikit learn

Main entry points:
- compute_embedding(dist_matrix, cfg)
- build_embedding_dataframe(dist_matrix, index, cfg)
"""

from typing import Iterable

import logging
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

from .config import AffinitreeConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core embedding
# ---------------------------------------------------------------------------


def _embedding_method(cfg: AffinitreeConfig) -> str:
    """
    Resolve the embedding method name from the configuration.

    AffinitreeConfig may define an `embedding_method` attribute. If it does not,
    we default to "mds". This keeps the module backward compatible with older
    configs that do not yet specify an embedding method.
    """
    method = getattr(cfg, "embedding_method", "mds")
    return str(method).lower()


def compute_embedding(
    dist_matrix: np.ndarray,
    cfg: AffinitreeConfig,
) -> np.ndarray:
    """
    Compute a 2D embedding from a pairwise distance matrix.

    Parameters
    ----------
    dist_matrix
        2D numpy array of shape (n_samples, n_samples) with pairwise distances.
        Diagonal entries are assumed to be zero.
    cfg
        AffinitreeConfig with random_state and optional embedding_method.

    Returns
    -------
    coords : np.ndarray
        Array of shape (n_samples, 2) with x and y coordinates.
    """
    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square 2D array")

    n_samples = dist_matrix.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot compute embedding for zero samples")

    method = _embedding_method(cfg)
    logger.info(
        "Computing %s embedding for %d individuals",
        method.upper(),
        n_samples,
    )

    if method == "mds":
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=cfg.random_state,
            n_init=4,
            max_iter=300,
        )
        coords = mds.fit_transform(dist_matrix)
        # Not all versions expose stress_ as a public attribute, so guard the log
        stress = getattr(mds, "stress_", None)
        if stress is not None:
            logger.info("MDS embedding completed with stress %.4f", float(stress))
        else:
            logger.info("MDS embedding completed")
    else:
        raise ValueError(f"Unsupported embedding method '{method}'")

    if coords.shape != (n_samples, 2):
        raise RuntimeError(
            f"Expected embedding of shape ({n_samples}, 2) but got {coords.shape}"
        )

    return coords


# ---------------------------------------------------------------------------
# Helper to build a DataFrame
# ---------------------------------------------------------------------------


def build_embedding_dataframe(
    dist_matrix: np.ndarray,
    index: Iterable,
    cfg: AffinitreeConfig,
) -> pd.DataFrame:
    """
    Compute a 2D embedding and wrap it in a pandas DataFrame.

    Parameters
    ----------
    dist_matrix
        Square pairwise distance matrix of shape (n_samples, n_samples).
    index
        Iterable of labels to use as the DataFrame index. Should have length
        n_samples and usually comes from df_norm.index or the merged scores
        DataFrame.
    cfg
        AffinitreeConfig with random_state and optional embedding_method.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ["x", "y"] and the given index.
    """
    coords = compute_embedding(dist_matrix, cfg)
    index_list = list(index)

    if len(index_list) != coords.shape[0]:
        raise ValueError(
            "Index length does not match number of rows in embedding "
            f"(len(index)={len(index_list)}, n_samples={coords.shape[0]})"
        )

    df_embed = pd.DataFrame(
        coords,
        columns=["x", "y"],
        index=index_list,
    )

    logger.info(
        "Built embedding DataFrame with %d rows",
        len(df_embed),
    )

    return df_embed