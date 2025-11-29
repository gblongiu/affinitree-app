"""
Affinitree package.

This package contains the core data pipeline and visualization logic for the
Affinitree project. It is organized into small modules for:

- loading and merging TCI score tables (data_io)
- preprocessing and normalization (preprocessing)
- distance and similarity calculations (similarity)
- 2D embedding of individuals (embedding)
- graph construction and clustering (graph)
- radial trait chart generation (radial_chart)
- Plotly figure construction (figure)
- building a complete HTML document for the visualization (html_builder)

The main public entry point is `build_affinitree_html`, which runs the full
pipeline and returns an HTML string ready to write to index.html.
"""

from .html_builder import build_affinitree_html

__all__ = ["build_affinitree_html"]