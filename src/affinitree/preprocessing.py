from __future__ import annotations

"""
Preprocessing utilities for the Affinitree project.

This module is responsible for:

- Selecting the subset of columns used as traits
- Converting raw score columns to numeric values
- Handling missing or invalid values in trait columns
- Normalizing traits so they can be used in distance calculations

The main public entry point is `build_trait_matrix`, which takes a merged
scores DataFrame and an AffinitreeConfig, and returns a numeric matrix of
normalized trait values aligned with the original index.
"""

from typing import List, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import AffinitreeConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column selection and basic cleaning
# ---------------------------------------------------------------------------


def trait_column_name_from_code(code: str) -> str:
    """
    Map a trait code (e.g. 'NS') to the corresponding column name in the data.

    By default this assumes columns are named like '<CODE> Total', e.g.:

        NS -> "NS Total"
        HA -> "HA Total"

    If your data uses a different naming convention you can either:

    - change this function, or
    - add an explicit mapping in the configuration and call that instead.
    """
    return f"{code} Total"


def get_trait_columns(cfg: AffinitreeConfig) -> List[str]:
    """
    Return the list of column names in the score table that should be used
    for distance calculations based on trait codes in the configuration.

    This uses `trait_column_name_from_code` to map from codes to column names.
    """
    cols = [trait_column_name_from_code(c) for c in cfg.distance_trait_codes]
    logger.debug("Using trait columns for distance: %s", ", ".join(cols))
    return cols


def coerce_traits_to_numeric(
    df: pd.DataFrame,
    trait_columns: List[str],
) -> pd.DataFrame:
    """
    Ensure that trait columns are numeric, coercing invalid values to NaN.

    Parameters
    ----------
    df
        Input DataFrame with raw score columns.
    trait_columns
        List of column names expected to contain trait scores.

    Returns
    -------
    DataFrame
        A copy of df with trait columns converted to numeric dtype.
    """
    df = df.copy()

    for col in trait_columns:
        if col not in df.columns:
            logger.warning("Expected trait column '%s' not found in DataFrame", col)
            df[col] = pd.NA
            continue

        before_invalid = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_invalid = df[col].isna().sum()

        if after_invalid > before_invalid:
            logger.warning(
                "Column '%s': coerced %d additional non-numeric values to NaN",
                col,
                after_invalid - before_invalid,
            )

    return df


def drop_rows_with_missing_traits(
    df: pd.DataFrame,
    trait_columns: List[str],
) -> Tuple[pd.DataFrame, int]:
    """
    Drop rows that have missing values in any of the specified trait columns.

    Returns a new DataFrame and the number of dropped rows.
    """
    df = df.copy()
    mask_valid = df[trait_columns].notna().all(axis=1)
    dropped = int((~mask_valid).sum())

    if dropped > 0:
        logger.warning(
            "Dropping %d rows with missing values in trait columns", dropped
        )

    return df.loc[mask_valid].copy(), dropped


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_traits(
    df: pd.DataFrame,
    trait_columns: List[str],
) -> pd.DataFrame:
    """
    Normalize trait columns using MinMax scaling to [0, 1].

    Parameters
    ----------
    df
        DataFrame with numeric trait columns and no missing values.
    trait_columns
        Column names to include in normalization.

    Returns
    -------
    DataFrame
        A new DataFrame with the same index and trait_columns, where each
        column has been scaled to the [0, 1] range.
    """
    if not trait_columns:
        raise ValueError("No trait columns provided for normalization")

    scaler = MinMaxScaler()
    values = df[trait_columns].to_numpy(dtype=float)

    if values.size == 0:
        raise ValueError("No rows available for normalization")

    scaled = scaler.fit_transform(values)

    df_norm = pd.DataFrame(
        scaled,
        columns=trait_columns,
        index=df.index,
    )

    logger.info(
        "Normalized %d trait columns for %d rows using MinMaxScaler",
        len(trait_columns),
        len(df_norm),
    )

    return df_norm


# ---------------------------------------------------------------------------
# High level entry point
# ---------------------------------------------------------------------------


def build_trait_matrix(
    df_scores: pd.DataFrame,
    cfg: AffinitreeConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the normalized trait matrix used for distance calculations.

    This function:

    1. Selects the trait columns based on config distance_trait_codes
    2. Coerces those columns to numeric
    3. Drops rows with missing trait values
    4. Applies MinMax scaling to [0, 1]

    Parameters
    ----------
    df_scores
        Merged scores DataFrame from `data_io.load_merged_scores`.
    cfg
        AffinitreeConfig that defines which traits to use.

    Returns
    -------
    (df_norm, trait_columns)
        df_norm is a DataFrame of shape (n_samples, n_traits) with normalized
        values. Its index aligns with a subset of df_scores index.
        trait_columns is the ordered list of trait column names used.
    """
    trait_columns = get_trait_columns(cfg)

    # Coerce to numeric and handle invalids
    df_numeric = coerce_traits_to_numeric(df_scores, trait_columns)

    # Drop rows with missing trait values
    df_clean, dropped = drop_rows_with_missing_traits(df_numeric, trait_columns)
    if dropped:
        logger.info("Dropped %d rows with incomplete trait data", dropped)

    # Normalize to [0, 1]
    df_norm = normalize_traits(df_clean, trait_columns)

    # Sanity check: no NaNs should remain
    if df_norm.isna().any().any():
        n_bad = int(df_norm.isna().sum().sum())
        logger.error("Normalized trait matrix still contains %d NaN values", n_bad)
        raise RuntimeError("NaNs found in normalized trait matrix")

    return df_norm, trait_columns