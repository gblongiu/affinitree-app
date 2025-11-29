from __future__ import annotations

"""
Radial trait chart generation for the Affinitree project.

This module is responsible for turning a single individual's TCI trait scores
into a radial (polar) bar chart image that can be embedded in the web
visualization.

Responsibilities:
- Take a row of trait scores from the merged scores DataFrame
- Select the traits used for the radial chart based on configuration
- Build a Matplotlib polar bar chart with consistent colors and ordering
- Encode the figure as a base64 PNG string for use in HTML and Plotly

Main entry point:
- create_radial_chart_image(row, cfg) -> str
"""

import base64
import io
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import AffinitreeConfig, TraitMeta
from .preprocessing import trait_column_name_from_code

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_trait_values_for_row(
    row: pd.Series,
    cfg: AffinitreeConfig,
) -> Dict[str, float]:
    """
    Extract trait values for the configured radial traits from a single row.

    The configuration provides trait codes such as "NS" and "HA". This
    function maps those codes to column names (for example "NS Total"),
    pulls out the values from the row, and returns a mapping from code
    to numeric value. Missing or non numeric values are treated as zero.
    """
    values: Dict[str, float] = {}

    for code in cfg.radial_trait_codes:
        col_name = trait_column_name_from_code(code)

        if col_name not in row:
            logger.warning(
                "Row is missing expected trait column '%s' for radial chart",
                col_name,
            )
            raw_value = np.nan
        else:
            raw_value = row[col_name]

        try:
            val = float(raw_value)
        except (TypeError, ValueError):
            val = np.nan

        if np.isnan(val):
            logger.warning(
                "Trait '%s' (column '%s') has NaN or invalid value in row; using 0",
                code,
                col_name,
            )
            val = 0.0

        values[code] = val

    return values


def _normalize_trait_values(values: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize trait values to [0, 1] for radial plotting.

    This does a simple min max scaling across the values for a single
    individual. The relative shape is what matters for the radial chart.

    If all values are zero, the function returns zeros for all traits.
    """
    if not values:
        return {}

    arr = np.array(list(values.values()), dtype=float)

    vmin = float(arr.min())
    vmax = float(arr.max())

    if vmax <= 0.0:
        # All zero or negative. Use zeros so the chart does not blow up.
        return {k: 0.0 for k in values.keys()}

    if np.isclose(vmax, vmin):
        # All values are equal but non zero. Map them all to 1.0.
        return {k: 1.0 for k in values.keys()}

    norm = (arr - vmin) / (vmax - vmin)

    return {
        code: float(val)
        for code, val in zip(values.keys(), norm)
    }


def _build_angles(n: int) -> np.ndarray:
    """
    Build equally spaced angles around the circle for n traits.

    The last point is not repeated so the bars tile the full 360 degrees
    without overlap.
    """
    if n <= 0:
        raise ValueError("Number of traits for radial chart must be positive")

    return np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)


def _colors_for_codes(
    codes: List[str],
    cfg: AffinitreeConfig,
) -> List[str]:
    """
    Return the color for each trait code based on configuration.

    If a code is not found in the config traits list, a neutral gray is used
    and a warning is logged.
    """
    meta_by_code = cfg.trait_meta_by_code()
    colors: List[str] = []

    for code in codes:
        meta: TraitMeta | None = meta_by_code.get(code)
        if meta is None:
            logger.warning(
                "No TraitMeta found for radial trait code '%s'; using gray",
                code,
            )
            colors.append("#888888")
        else:
            colors.append(meta.color)

    return colors


def _group_codes_by_kind(
    codes: List[str],
    cfg: AffinitreeConfig,
) -> Dict[str, List[str]]:
    """
    Group trait codes into temperament vs character based on TraitMeta.kind.

    Returns a mapping from kind to list of codes of that kind that are
    present in the given codes in order.
    """
    meta_by_code = cfg.trait_meta_by_code()
    groups: Dict[str, List[str]] = {"temperament": [], "character": []}

    for code in codes:
        meta: TraitMeta | None = meta_by_code.get(code)
        if meta is None:
            continue
        kind = meta.kind.lower()
        if kind not in groups:
            groups[kind] = []
        groups[kind].append(code)

    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_radial_chart_image(
    row: pd.Series,
    cfg: AffinitreeConfig,
    dpi: int = 120,
) -> str:
    """
    Create a radial bar chart for a single individual and return it as base64.

    Parameters
    ----------
    row
        A single row from the merged scores DataFrame. Must contain the trait
        columns that correspond to cfg.radial_trait_codes.
    cfg
        AffinitreeConfig that defines which traits to plot and their colors.
    dpi
        Dots per inch for the generated PNG. Higher values produce sharper
        images at the cost of file size.

    Returns
    -------
    str
        A base64 encoded PNG image string suitable for embedding in HTML.
        You can use it in an <img> tag as:
            <img src="data:image/png;base64, {{ base64_string }}" />
    """
    # Extract and normalize trait values
    raw_values = _get_trait_values_for_row(row, cfg)
    norm_values = _normalize_trait_values(raw_values)

    codes_in_order = list(norm_values.keys())
    vals_in_order = np.array(list(norm_values.values()), dtype=float)

    n_traits = len(codes_in_order)
    if n_traits == 0:
        raise ValueError("No trait values available for radial chart")

    # Angles and colors
    angles = _build_angles(n_traits)
    colors = _colors_for_codes(codes_in_order, cfg)

    # Set up Matplotlib figure
    fig, ax = plt.subplots(
        subplot_kw={"projection": "polar"},
        figsize=(4, 4),
        dpi=dpi,
    )

    # Bars from center to value
    width = 2.0 * np.pi / n_traits
    bars = ax.bar(
        angles,
        vals_in_order,
        width=width * 0.9,
        bottom=0.0,
        align="center",
        color=colors,
        edgecolor="white",
        linewidth=1.0,
    )

    # Labels at the end of bars
    for angle, value, code in zip(angles, vals_in_order, codes_in_order):
        # Small radial offset to place label just outside bar
        r_label = value + 0.12
        ax.text(
            angle,
            r_label,
            code,
            ha="center",
            va="center",
            fontsize=9,
        )

    # Style tweaks: start at the top, rotate clockwise
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    # Remove radial grid and labels for a cleaner look
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

    # Build legend entries for temperament vs character
    groups = _group_codes_by_kind(codes_in_order, cfg)
    meta_by_code = cfg.trait_meta_by_code()

    legend_handles = []
    legend_labels = []

    # Temperament
    if groups.get("temperament"):
        # Use the color of the first temperament code as representative
        first_code = groups["temperament"][0]
        color = meta_by_code.get(first_code, TraitMeta(first_code, "", "", "#888888", 1.0)).color
        handle_temp = plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
        )
        legend_handles.append(handle_temp)
        legend_labels.append("Temperament traits")

    # Character
    if groups.get("character"):
        first_code = groups["character"][0]
        color = meta_by_code.get(first_code, TraitMeta(first_code, "", "", "#888888", 1.0)).color
        handle_char = plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
        )
        legend_handles.append(handle_char)
        legend_labels.append("Character traits")

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(1.3, 1.1),
            frameon=False,
            fontsize=9,
        )

    plt.tight_layout()

    # Encode figure to base64 PNG
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buffer.seek(0)

    img_bytes = buffer.read()
    base64_str = base64.b64encode(img_bytes).decode("ascii")

    logger.info("Generated radial chart image for one individual")

    return base64_str