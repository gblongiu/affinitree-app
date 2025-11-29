from __future__ import annotations

"""
Data loading utilities for the Affinitree project.

This module is responsible for reading TCI score tables from disk and returning
a single pandas DataFrame that the rest of the pipeline can consume.

Responsibilities:
- Load the base TCI scores table
- Optionally load a user scores table
- Concatenate base + user data with consistent columns
- Add simple metadata columns to help distinguish sources if needed

All file paths and high level settings are taken from AffinitreeConfig in
config.py. Call `load_merged_scores(config)` as the main entry point.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import logging
import pandas as pd

from .config import AffinitreeConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedTables:
    """Container for the raw base and user tables before merging."""

    base: pd.DataFrame
    user: Optional[pd.DataFrame]


# ---------------------------------------------------------------------------
# Low level loaders
# ---------------------------------------------------------------------------


def _load_table_any(path: Path) -> pd.DataFrame:
    """
    Load a table from an Excel or CSV file based on its extension.

    Supported formats:
      - .xlsx / .xls via pandas.read_excel
      - .csv via pandas.read_csv

    Raises FileNotFoundError if the file does not exist, and ValueError for
    unsupported suffixes.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        logger.info("Loading Excel table from %s", path)
        return pd.read_excel(path)
    if suffix == ".csv":
        logger.info("Loading CSV table from %s", path)
        return pd.read_csv(path)

    raise ValueError(f"Unsupported data file extension '{suffix}' for {path}")


def load_base_table(cfg: AffinitreeConfig) -> pd.DataFrame:
    """
    Load the base TCI scores table defined in the configuration.

    This should contain the canonical set of TCI scores (e.g. Neuma_TCI_Score).
    """
    path = cfg.base_data_path
    df = _load_table_any(path)

    logger.info("Loaded base table with %d rows and %d columns from %s",
                len(df), df.shape[1], path)

    return df


def load_user_table(cfg: AffinitreeConfig) -> Optional[pd.DataFrame]:
    """
    Load the optional user scores table if it exists.

    If the configured path is None or the file does not exist, this returns
    None and logs at INFO level.
    """
    path = cfg.user_data_path
    if path is None:
        logger.info("No user data path configured; skipping user table load")
        return None

    if not path.exists():
        logger.info("User data path %s does not exist; skipping user table", path)
        return None

    df = _load_table_any(path)

    logger.info("Loaded user table with %d rows and %d columns from %s",
                len(df), df.shape[1], path)

    return df


def load_tables(cfg: AffinitreeConfig) -> LoadedTables:
    """
    Load both base and user tables according to the configuration.

    Returns
    -------
    LoadedTables
        A simple container with .base and .user attributes. The .user attribute
        may be None if there is no user data.
    """
    base_df = load_base_table(cfg)
    user_df = load_user_table(cfg)
    return LoadedTables(base=base_df, user=user_df)


# ---------------------------------------------------------------------------
# Merging and alignment
# ---------------------------------------------------------------------------


def _align_columns(base: pd.DataFrame, user: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align columns between base and user tables before concatenation.

    - Columns present in base but missing in user are added to user with NaN.
    - Columns present in user but not base are kept, but a warning is logged.
      (You can adjust this behavior if you prefer to drop them.)

    Returns
    -------
    (base_aligned, user_aligned)
        Two DataFrames that share the same column set.
    """
    base_cols = set(base.columns)
    user_cols = set(user.columns)

    missing_in_user = base_cols - user_cols
    extra_in_user = user_cols - base_cols

    if missing_in_user:
        logger.warning(
            "User table is missing %d columns present in base: %s",
            len(missing_in_user),
            ", ".join(sorted(missing_in_user)),
        )
        for col in missing_in_user:
            user[col] = pd.NA

    if extra_in_user:
        logger.warning(
            "User table has %d extra columns not in base: %s",
            len(extra_in_user),
            ", ".join(sorted(extra_in_user)),
        )
        # We keep them by default; you could also choose to drop them:
        # user = user.drop(columns=list(extra_in_user))

    # Reorder user columns to match base where possible, then append extras.
    # This ensures a stable and predictable column order.
    ordered_cols = list(base.columns) + [c for c in user.columns if c not in base.columns]
    base = base.reindex(columns=ordered_cols, fill_value=pd.NA)
    user = user.reindex(columns=ordered_cols, fill_value=pd.NA)

    return base, user


def merge_base_and_user(tables: LoadedTables) -> pd.DataFrame:
    """
    Merge the base and user tables into a single DataFrame.

    A new column `source` is added with values:
      - "base" for rows from the base table
      - "user" for rows from the user table (if any)

    Parameters
    ----------
    tables
        The LoadedTables object returned by load_tables.

    Returns
    -------
    pandas.DataFrame
        Combined table with a `source` column.
    """
    base_df = tables.base.copy()
    base_df["source"] = "base"

    if tables.user is None or tables.user.empty:
        logger.info("No user table to merge; returning base table only")
        return base_df

    user_df = tables.user.copy()
    user_df["source"] = "user"

    # Align column sets before concatenation
    base_aligned, user_aligned = _align_columns(base_df, user_df)

    merged = pd.concat([base_aligned, user_aligned], ignore_index=True)

    logger.info(
        "Merged base (%d rows) and user (%d rows) into %d total rows",
        len(base_df),
        len(user_df),
        len(merged),
    )

    return merged


def load_merged_scores(cfg: AffinitreeConfig) -> pd.DataFrame:
    """
    High level entry point: load base and user tables and return a merged frame.

    This is the function the rest of the pipeline should call.

    Parameters
    ----------
    cfg
        AffinitreeConfig instance specifying data paths and options.

    Returns
    -------
    pandas.DataFrame
        Combined score table with a `source` column indicating origin.
    """
    tables = load_tables(cfg)
    merged = merge_base_and_user(tables)

    logger.info(
        "Final merged scores table has %d rows and %d columns",
        len(merged),
        merged.shape[1],
    )

    return merged