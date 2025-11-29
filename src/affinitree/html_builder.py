from __future__ import annotations

"""
HTML builder for the Affinitree project.

This module provides the top level pipeline that:

1. Loads and merges TCI score tables
2. Builds a normalized trait matrix
3. Computes a pairwise distance matrix
4. Computes a 2D embedding
5. Builds a graph and assigns roles
6. Constructs a Plotly figure
7. Wraps the figure JSON into a standalone HTML document string

Main public entry point:
    build_affinitree_html(cfg: AffinitreeConfig | None = None) -> str
"""

import json
import logging
from typing import Optional

import pandas as pd

from .config import AffinitreeConfig
from .data_io import load_merged_scores
from .preprocessing import build_trait_matrix
from .similarity import build_distance_matrix
from .embedding import build_embedding_dataframe
from .figure import build_plotly_figure
from . import graph as graph_mod  # expect graph_mod.build_graph(...)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <!-- Plotly from CDN -->
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <!-- Affinitree client script -->
  <script src="web/static/js/affinitree.js"></script>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                   sans-serif;
      background: #0b1020;
      color: #f5f5f5;
    }}
    #root {{
      display: flex;
      flex-direction: row;
      height: 100vh;
      width: 100%;
      box-sizing: border-box;
    }}
    #affinitree-plot {{
      flex: 3;
      min-width: 0;
    }}
    #sidebar {{
      flex: 1;
      min-width: 260px;
      max-width: 420px;
      border-left: 1px solid #222;
      box-sizing: border-box;
      padding: 14px;
      background: #101528;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    #sidebar h1 {{
      margin: 0 0 4px 0;
      font-size: 18px;
      font-weight: 600;
    }}
    #sidebar .subtitle {{
      font-size: 13px;
      color: #c0c3d6;
      margin-bottom: 8px;
    }}
    #radial-chart-container {{
      flex: 1;
      border-radius: 8px;
      background: #050814;
      border: 1px solid #20253a;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      padding: 8px;
    }}
    #radial-chart-container img {{
      max-width: 100%;
      height: auto;
      display: block;
    }}
    #sidebar footer {{
      font-size: 11px;
      color: #8a8ea5;
      margin-top: 6px;
    }}
  </style>
</head>
<body>
  <div id="root">
    <div id="affinitree-plot"></div>
    <aside id="sidebar">
      <header>
        <h1>Affinitree</h1>
        <div class="subtitle">
          Click a node to inspect temperament and character shape
        </div>
      </header>
      <section id="radial-chart-container">
        <span style="font-size: 13px; color: #7d8197;">
          Select a node on the left to view its trait profile
        </span>
      </section>
      <footer>
        <div>Data: Temperament and Character Inventory (TCI)</div>
        <div>Visualization built with Plotly and Affinitree</div>
      </footer>
    </aside>
  </div>

  <!-- Serialized Plotly figure -->
  <script id="plot-data" type="application/json">
{plot_json}
  </script>

  <!-- The affinitree.js script is expected to read #plot-data and initialize the plot -->
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def _run_pipeline(cfg: AffinitreeConfig) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Run the full data to figure pipeline.

    Returns
    -------
    (df_scores, df_embed, fig)
        df_scores: merged scores DataFrame with any role metadata
        df_embed:  DataFrame with x and y coordinates
        fig:       Plotly Figure
    """
    # 1. Load and merge score tables
    df_scores = load_merged_scores(cfg)
    logger.info("Pipeline: loaded merged scores with %d rows", len(df_scores))

    # 2. Build normalized trait matrix
    df_norm, trait_columns = build_trait_matrix(df_scores, cfg)
    logger.info(
        "Pipeline: built trait matrix with %d rows and %d traits",
        len(df_norm),
        len(trait_columns),
    )

    # 3. Distance matrix
    dist_matrix = build_distance_matrix(df_norm, trait_columns, cfg)
    logger.info("Pipeline: distance matrix shape %s", dist_matrix.shape)

    # 4. Embedding
    df_embed = build_embedding_dataframe(dist_matrix, df_norm.index, cfg)
    logger.info("Pipeline: embedding frame shape %s", df_embed.shape)

    # 5. Graph construction and roles
    # We expect graph_mod.build_graph to accept
    # (df_scores, df_embed, dist_matrix, cfg) and return a networkx Graph
    G = graph_mod.build_graph(df_scores, df_embed, dist_matrix, cfg)
    logger.info(
        "Pipeline: built graph with %d nodes and %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )

    # 6. Plotly figure
    fig = build_plotly_figure(G, df_embed, df_scores, cfg)

    return df_scores, df_embed, fig


def _figure_to_html(fig, cfg: AffinitreeConfig) -> str:
    """
    Serialize a Plotly figure and wrap it in the HTML template.
    """
    plot_json_str = fig.to_json()

    # Indent for readability inside the script tag
    plot_json_indented = "\n".join(
        "    " + line for line in plot_json_str.splitlines()
    )

    html = _HTML_TEMPLATE.format(
        title=cfg.plot_title,
        plot_json=plot_json_indented,
    )

    return html


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_affinitree_html(cfg: Optional[AffinitreeConfig] = None) -> str:
    """
    Run the full Affinitree pipeline and return a standalone HTML document
    as a string.

    Parameters
    ----------
    cfg
        Optional AffinitreeConfig. If None, a default config is created.

    Returns
    -------
    str
        Complete HTML suitable for writing directly to an index.html file.
    """
    if cfg is None:
        cfg = AffinitreeConfig()

    logger.info("Starting Affinitree pipeline with config: %s", cfg)

    _, _, fig = _run_pipeline(cfg)
    html = _figure_to_html(fig, cfg)

    logger.info("Affinitree HTML build complete")

    return html