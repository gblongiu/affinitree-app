#!/usr/bin/env python3
"""Maintain the research section in the existing Affinitree README.

The updater preserves the surrounding project documentation and adds the
research link and conceptual lineage without replacing unrelated sections.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
HEADING = "## Exploratory cohort findings"
PRESENTATION_LINK = "[Research presentation](https://gblongiu.github.io/affinitree-app/)"
LINEAGE = (
    "**Design lineage:** Affinitree captures a single TCI profile as a fixed point "
    "in a similarity map. [Rhizome](https://gblongiu.github.io/rhizome-living-network/) "
    "develops the idea toward relationships and positions that may change over time. "
    "Its evolving-network scenes are illustrative; the cohort study contains no "
    "longitudinal observations. The seven colors retain their original trait meanings: "
    "Novelty Seeking (orange), Harm Avoidance (red), Reward Dependence (pink), "
    "Persistence (brown), Self-Directedness (blue), Cooperativeness (green), "
    "and Self-Transcendence (purple)."
)
SECTION = f"""
## Exploratory cohort findings

{PRESENTATION_LINK} · [Analysis implementation](scripts/research_analysis.py)

{LINEAGE}

I analyzed the 50 baseline score profiles in `data/Neuma_TCI_Score.xlsx`, aggregating 29 subscales into seven TCI composite dimensions. This cohort yields **1,225 distinct profile pairs** for distance-based comparison.

- **Trait relationships:** Nine of 21 exploratory Pearson correlations remained below a 5% false-discovery-rate threshold after Benjamini–Hochberg adjustment. Persistence and Self-Transcendence showed the strongest observed association (*r* = 0.630).
- **Dimensionality reduction:** Two-dimensional metric MDS preserved much of the rank ordering of seven-dimensional Euclidean distances (Spearman *ρ* = 0.899), with normalized Stress-1 = 0.203.
- **Neighborhood preservation:** The 2D projection retained 54.4% of the original five-nearest-neighbor connections.
- **Clustering diagnostic:** Four-group Ward clustering gave a silhouette score of 0.160, compared with 0.241 for a two-group solution. These clusters should not be interpreted as validated psychological types.

The analysis is **exploratory**. The 50 profiles are not a representative population sample, subscale aggregation has not been independently validated against the instrument's scoring manual, and correlations do not establish causal or diagnostic claims. Questionnaire submissions are not included in these figures.

To regenerate the published aggregate statistics and website:

```bash
python scripts/research_analysis.py
python -m unittest discover -s tests -p 'test_research_analysis.py' -v
python -m http.server 8000 --directory docs
```

The local preview is available at `http://localhost:8000/`. The website is a standalone static presentation and does not depend on the questionnaire backend. GitHub Pages can serve it directly from the `docs/` folder on `main`.
""".strip()


def main() -> None:
    if not README.exists():
        raise SystemExit(f"README.md was not found at {README}")
    original = README.read_text(encoding="utf-8")
    if HEADING in original:
        if LINEAGE in original:
            print("Research section and design lineage are already present.")
            return
        if PRESENTATION_LINK in original:
            revised = original.replace(PRESENTATION_LINK, PRESENTATION_LINK + "\n\n" + LINEAGE, 1)
            README.write_text(revised, encoding="utf-8")
            print("Updated the existing research section with the project lineage.")
        else:
            print("Found a customized research section; preserved it without changes.")
        return
    divider = "\n---\n"
    if divider in original:
        revised = original.replace(divider, "\n\n" + SECTION + "\n\n---\n", 1)
    else:
        revised = original.rstrip() + "\n\n" + SECTION + "\n"
    README.write_text(revised, encoding="utf-8")
    print("Added the exploratory research section and design lineage.")


if __name__ == "__main__":
    main()
