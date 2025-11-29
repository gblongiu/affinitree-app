from __future__ import annotations

"""
Configuration and shared metadata for the Affinitree project.

This module defines:

- Paths to data files relative to the project root
- Trait metadata for temperament and character
- Default pipeline parameters such as distance metric and number of clusters

Most code in the package should import configuration values from here rather
than hard coding paths or constants.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root is two levels up from this file.
#   project_root/
#     src/
#       affinitree/
#         config.py
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"

# Default data file locations
DEFAULT_BASE_DATA_PATH: Path = DATA_DIR / "Neuma_TCI_Score.xlsx"
DEFAULT_USER_DATA_PATH: Path = DATA_DIR / "Neuma_TCI_Score_user.xlsx"

# Questionnaire source and parsed JSON
DEFAULT_QUESTIONNAIRE_TEXT_PATH: Path = DATA_DIR / "fullQuestionnare.txt"
DEFAULT_QUESTIONNAIRE_JSON_PATH: Path = DATA_DIR / "questions.json"


# ---------------------------------------------------------------------------
# Trait metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraitMeta:
    """Metadata for a single TCI trait used in visualization and analysis."""

    code: str               # Short code such as NS, HA, P
    name: str               # Human friendly name
    kind: str               # temperament or character
    color: str              # Hex color code for visualization
    distance_weight: float  # Weight used in distance calculations


# Core set of traits used in the radial chart and summaries
TRAITS: List[TraitMeta] = [
    TraitMeta(
        code="P",
        name="Persistence",
        kind="temperament",
        color="#8B4513",      # brown
        distance_weight=1.0,
    ),
    TraitMeta(
        code="HA",
        name="Harm Avoidance",
        kind="temperament",
        color="#FF0000",      # red
        distance_weight=1.0,
    ),
    TraitMeta(
        code="NS",
        name="Novelty Seeking",
        kind="temperament",
        color="#FFA500",      # orange
        distance_weight=1.0,
    ),
    TraitMeta(
        code="RD",
        name="Reward Dependence",
        kind="temperament",
        color="#FF69B4",      # pink
        distance_weight=1.0,
    ),
    TraitMeta(
        code="CO",
        name="Cooperativeness",
        kind="character",
        color="#4CAF50",      # green
        distance_weight=1.0,
    ),
    TraitMeta(
        code="ST",
        name="Self Transcendence",
        kind="character",
        color="#7E2F94",      # purple
        distance_weight=1.0,
    ),
    TraitMeta(
        code="SD",
        name="Self Directedness",
        kind="character",
        color="#0000FF",      # blue
        distance_weight=1.0,
    ),
]

# Convenience lookups
TRAIT_BY_CODE: Dict[str, TraitMeta] = {t.code: t for t in TRAITS}


def trait_codes_for_distance() -> List[str]:
    """
    Return the list of trait codes used in distance calculations.

    For now this uses all core traits. You can shrink or reorder this list
    if you want to base distance on a subset of traits.
    """
    return [t.code for t in TRAITS]


def trait_codes_for_radial_chart() -> List[str]:
    """
    Return the list of trait codes used in the radial chart.

    This usually matches the distance traits, but you can diverge if needed.
    """
    return [t.code for t in TRAITS]


# ---------------------------------------------------------------------------
# Clustering and roles
# ---------------------------------------------------------------------------

# Default number of clusters for the hierarchical clustering step
DEFAULT_N_CLUSTERS: int = 4

# Mapping from cluster index to human friendly role label
DEFAULT_CLUSTER_LABELS: Dict[int, str] = {
    0: "Root",
    1: "Trunk",
    2: "Branch",
    3: "Leaf",
}


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class AffinitreeConfig:
    """
    Top level configuration object for the Affinitree pipeline.

    Pass this into the main pipeline functions so that tests, scripts, and
    the Flask app can reuse the same settings.
    """

    base_data_path: Path = DEFAULT_BASE_DATA_PATH
    user_data_path: Optional[Path] = DEFAULT_USER_DATA_PATH

    questionnaire_text_path: Path = DEFAULT_QUESTIONNAIRE_TEXT_PATH
    questionnaire_json_path: Path = DEFAULT_QUESTIONNAIRE_JSON_PATH

    # Trait configuration
    traits: List[TraitMeta] = field(default_factory=lambda: TRAITS.copy())
    distance_trait_codes: List[str] = field(
        default_factory=trait_codes_for_distance
    )
    radial_trait_codes: List[str] = field(
        default_factory=trait_codes_for_radial_chart
    )

    # Distance and embedding parameters
    distance_metric: str = "euclidean"
    random_state: int = 42

    # Clustering
    n_clusters: int = DEFAULT_N_CLUSTERS
    cluster_labels: Dict[int, str] = field(
        default_factory=lambda: DEFAULT_CLUSTER_LABELS.copy()
    )

    # Plot and HTML options
    plot_title: str = "Affinitree"
    show_legend: bool = False

    def trait_meta_by_code(self) -> Dict[str, TraitMeta]:
        """Return a mapping from trait code to TraitMeta for this config."""
        return {t.code: t for t in self.traits}