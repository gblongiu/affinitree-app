from __future__ import annotations

"""
Plotly figure construction for the Affinitree project.

This module builds the main interactive Plotly figure that shows

- a network of individuals positioned in 2D space
- edges representing connections or similarity links
- node markers colored by role
- per node radial charts attached as base64 images in customdata

The main entry point is `build_plotly_figure`, which takes

- a networkx graph
- an embedding DataFrame with x and y coordinates
- the merged scores DataFrame
- an AffinitreeConfig instance

and returns a Plotly Figure ready for serialization or rendering.
"""

from typing import Dict, Any

import logging
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from .config import AffinitreeConfig
from .radial_chart import create_radial_chart_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _role_color(role: str) -> str:
    """
    Map a role label to a hex color.

    You can adjust this mapping if you change the cluster labels or
    want a different color palette.
    """
    mapping = {
        "Root": "#1f77b4",
        "Trunk": "#2ca02c",
        "Branch": "#ff7f0e",
        "Leaf": "#9467bd",
    }
    return mapping.get(role, "#7f7f7f")


def _build_edge_trace(
    G: nx.Graph,
    df_embed: pd.DataFrame,
) -> go.Scatter:
    """
    Build a Plotly Scatter trace for graph edges.

    Edges are drawn as line segments between embedded node positions.
    """
    edge_x = []
    edge_y = []

    for u, v in G.edges():
        if u not in df_embed.index or v not in df_embed.index:
            logger.debug(
                "Skipping edge (%r, %r) because one endpoint is missing in embedding",
                u,
                v,
            )
            continue

        x0 = float(df_embed.loc[u, "x"])
        y0 = float(df_embed.loc[u, "y"])
        x1 = float(df_embed.loc[v, "x"])
        y1 = float(df_embed.loc[v, "y"])

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(150,150,150,0.45)"),
        hoverinfo="skip",
        name="connections",
    )

    return edge_trace


def _build_node_trace(
    G: nx.Graph,
    df_embed: pd.DataFrame,
    df_scores: pd.DataFrame,
    cfg: AffinitreeConfig,
) -> go.Scatter:
    """
    Build a Plotly Scatter trace for graph nodes.

    Each node gets

    - x, y coordinates from df_embed
    - hover text with basic details
    - marker color based on role
    - customdata entry with base64 encoded radial chart image
    """
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_images = []

    # Optional columns we may use for hover text
    has_name_col = "Name" in df_scores.columns
    has_role_col = "Role" in df_scores.columns
    has_source_col = "source" in df_scores.columns

    for node in G.nodes():
        if node not in df_embed.index:
            logger.debug(
                "Skipping node %r because it is missing in embedding index", node
            )
            continue
        if node not in df_scores.index:
            logger.debug(
                "Skipping node %r because it is missing in scores index", node
            )
            continue

        x = float(df_embed.loc[node, "x"])
        y = float(df_embed.loc[node, "y"])
        row = df_scores.loc[node]

        # Derive display label, role, and source
        label = str(row["Name"]) if has_name_col else str(node)
        role = str(row["Role"]) if has_role_col else "Unlabeled"
        source = str(row["source"]) if has_source_col else "base"

        node_x.append(x)
        node_y.append(y)
        node_color.append(_role_color(role))

        # Build hover text
        hover_parts = [label]
        if has_role_col:
            hover_parts.append(f"Role: {role}")
        if has_source_col:
            hover_parts.append(f"Source: {source}")
        hover_text = "<br>".join(hover_parts)
        node_text.append(hover_text)

        # Generate radial chart image for this row
        try:
            base64_img = create_radial_chart_image(row, cfg)
            img_uri = f"data:image/png;base64,{base64_img}"
        except Exception as exc:
            logger.exception(
                "Failed to generate radial chart image for node %r, using empty image",
                node,
            )
            img_uri = ""

        node_images.append(img_uri)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=14,
            color=node_color,
            line=dict(width=1, color="#ffffff"),
        ),
        customdata=node_images,
        name="individuals",
    )

    logger.info(
        "Built node trace with %d nodes and custom images",
        len(node_x),
    )

    return node_trace


def _build_layout(cfg: AffinitreeConfig) -> go.Layout:
    """
    Build a Plotly Layout object for the Affinitree visualization.

    Layout is configured to be responsive and to keep the x and y scales
    locked so that the graph does not warp when the window resizes.
    """
    layout = go.Layout(
        title=dict(text=cfg.plot_title, x=0.5),
        showlegend=cfg.show_legend,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1.0,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        clickmode="event+select",
    )

    return layout


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_plotly_figure(
    G: nx.Graph,
    df_embed: pd.DataFrame,
    df_scores: pd.DataFrame,
    cfg: AffinitreeConfig,
) -> go.Figure:
    """
    Build the full Plotly figure for the Affinitree visualization.

    Parameters
    ----------
    G
        Networkx graph whose nodes correspond to individuals. Node identifiers
        must appear in both df_embed.index and df_scores.index.
    df_embed
        DataFrame with columns "x" and "y" giving 2D coordinates for each
        individual. Index must align with df_scores and graph nodes.
    df_scores
        Merged scores DataFrame with trait columns and optional metadata such
        as "Name", "Role", and "source".
    cfg
        AffinitreeConfig object with plot title and other options.

    Returns
    -------
    plotly.graph_objs.Figure
        Fully constructed figure with edge and node traces.
    """
    if "x" not in df_embed.columns or "y" not in df_embed.columns:
        raise ValueError("df_embed must contain 'x' and 'y' columns")

    logger.info(
        "Building Plotly figure for graph with %d nodes and %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )

    edge_trace = _build_edge_trace(G, df_embed)
    node_trace = _build_node_trace(G, df_embed, df_scores, cfg)
    layout = _build_layout(cfg)

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    logger.info("Plotly figure construction complete")

    return fig