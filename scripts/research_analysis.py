#!/usr/bin/env python3
"""Reproduce the exploratory trait analysis and its published summary page.

The analysis uses only the original baseline workbook. Questionnaire
submissions are deliberately excluded from the published cohort statistics.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
BASE_WORKBOOK = ROOT / "data" / "Neuma_TCI_Score.xlsx"
SITE = ROOT / "docs"

TRAITS = (
    ("NS", "Novelty Seeking", 4),
    ("HA", "Harm Avoidance", 4),
    ("RD", "Reward Dependence", 4),
    ("P", "Persistence", 4),
    ("SD", "Self-Directedness", 5),
    ("CO", "Cooperativeness", 5),
    ("ST", "Self-Transcendence", 3),
)


def composite_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Sum each trait's subscales into seven descriptive composite scores."""
    names = [f"{code}{part}" for code, _, count in TRAITS for part in range(1, count + 1)]
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing expected subscale columns: {', '.join(missing)}")

    components = frame.loc[:, names].apply(pd.to_numeric, errors="coerce")
    invalid = components.isna().sum().sum()
    if invalid:
        raise ValueError(f"Found {invalid} missing or nonnumeric subscale scores")
    if not np.isfinite(components.to_numpy(dtype=float)).all():
        raise ValueError("All subscale scores must be finite")
    if components.empty:
        raise ValueError("The baseline workbook contains no scored records")

    return pd.DataFrame(
        {
            code: components[[f"{code}{part}" for part in range(1, count + 1)]].sum(axis=1)
            for code, _, count in TRAITS
        },
        index=frame.index,
    )


def adjust_bh(p_values: list[float]) -> list[float]:
    """Apply Benjamini-Hochberg false-discovery-rate adjustment."""
    total = len(p_values)
    order = sorted(range(total), key=lambda index: p_values[index])
    adjusted = [1.0] * total
    smallest = 1.0
    for rank in range(total, 0, -1):
        index = order[rank - 1]
        smallest = min(smallest, p_values[index] * total / rank)
        adjusted[index] = float(smallest)
    return adjusted


def metric_mds(distances: np.ndarray) -> np.ndarray:
    """Fit metric SMACOF with the same explicit settings across sklearn versions."""
    options: dict[str, Any] = {
        "n_components": 2,
        "n_init": 4,
        "max_iter": 300,
        "eps": 1e-6,
        "random_state": 42,
    }
    supported = inspect.signature(MDS).parameters
    if "metric_mds" in supported:
        options.update(metric="precomputed", metric_mds=True, init="random")
    else:
        options.update(dissimilarity="precomputed", metric=True)
    return MDS(**options).fit_transform(distances)


def neighbor_retention(original: np.ndarray, projected: np.ndarray, k: int = 5) -> float:
    """Fraction of original k-nearest-neighbor links preserved in the map."""
    if len(original) <= k:
        raise ValueError("More records than the neighbor count are required")
    before = pairwise_distances(original)
    after = pairwise_distances(projected)
    np.fill_diagonal(before, np.inf)
    np.fill_diagonal(after, np.inf)
    common = sum(
        len(set(np.argsort(before[i])[:k]) & set(np.argsort(after[i])[:k]))
        for i in range(len(original))
    )
    return common / (len(original) * k)


def distance_bins(original: np.ndarray, embedded: np.ndarray, bins: int = 13) -> list[dict[str, float | int]]:
    """Summarize pairwise distances without publishing individual profiles."""
    order = np.argsort(original)
    groups = np.array_split(order, bins)
    return [
        {
            "original": round(float(np.median(original[group])), 4),
            "embedded": round(float(np.median(embedded[group])), 4),
            "pairs": int(len(group)),
        }
        for group in groups
        if len(group)
    ]


def analyze(scores: pd.DataFrame) -> dict[str, Any]:
    """Calculate descriptive, embedding, and cluster-quality statistics."""
    n = len(scores)
    if n < 10:
        raise ValueError("At least 10 records are required for this analysis")

    codes = [code for code, _, _ in TRAITS]
    matrix = scores.loc[:, codes].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Composite scores must contain only finite numbers")
    if (np.ptp(matrix, axis=0) <= 0).any():
        raise ValueError("Each trait must vary across the cohort")

    correlations = np.corrcoef(matrix, rowvar=False)
    tests = []
    pvals = []
    for i, j in combinations(range(len(codes)), 2):
        correlation, pvalue = pearsonr(matrix[:, i], matrix[:, j])
        tests.append({"a": codes[i], "b": codes[j], "r": round(float(correlation), 5)})
        pvals.append(float(pvalue))

    adjusted = adjust_bh(pvals)
    for test, original_p, qvalue in zip(tests, pvals, adjusted):
        test["p"] = round(original_p, 8)
        test["q"] = round(qvalue, 8)
        test["fdr_significant"] = bool(qvalue < 0.05)
    strongest = sorted(tests, key=lambda test: abs(test["r"]), reverse=True)

    scaled = MinMaxScaler().fit_transform(matrix)
    seven_d = pairwise_distances(scaled, metric="euclidean")
    projected = metric_mds(seven_d)
    two_d = pairwise_distances(projected, metric="euclidean")
    upper = np.triu_indices(n, 1)
    before = seven_d[upper]
    after = two_d[upper]

    stress_denominator = float(np.sum(after**2))
    if stress_denominator <= 0:
        raise ValueError("Cannot calculate normalized stress for a degenerate map")
    stress_1 = math.sqrt(float(np.sum((before - after) ** 2)) / stress_denominator)
    distance_pearson = pearsonr(before, after).statistic
    distance_spearman = spearmanr(before, after).statistic

    clusters: dict[str, Any] = {}
    for group_count in (2, 4):
        labels = AgglomerativeClustering(n_clusters=group_count, linkage="ward").fit_predict(scaled)
        counts = sorted((int(np.sum(labels == group)) for group in np.unique(labels)), reverse=True)
        clusters[str(group_count)] = {
            "silhouette": round(float(silhouette_score(scaled, labels)), 5),
            "sizes": counts,
        }

    return {
        "source": "data/Neuma_TCI_Score.xlsx",
        "sample": "Baseline workbook only; optional questionnaire submissions excluded",
        "analysis_type": "Exploratory cross-sectional analysis",
        "record_count": n,
        "subscale_count": sum(count for _, _, count in TRAITS),
        "trait_count": len(TRAITS),
        "traits": [
            {"code": code, "name": name, "subscales": count}
            for code, name, count in TRAITS
        ],
        "profile_pairs": n * (n - 1) // 2,
        "correlation_tests": len(tests),
        "fdr_significant": sum(test["fdr_significant"] for test in tests),
        "correlation_matrix": [
            [round(float(value), 5) for value in row] for row in correlations
        ],
        "correlations": tests,
        "strongest_pairs": strongest[:5],
        "projection": {
            "source_dimensions": len(TRAITS),
            "target_dimensions": 2,
            "method": "Metric SMACOF multidimensional scaling",
            "distance": "Euclidean distance after min-max scaling each composite trait",
            "distance_spearman": round(float(distance_spearman), 5),
            "distance_pearson": round(float(distance_pearson), 5),
            "stress_1": round(stress_1, 5),
            "nearest_five_retained": round(neighbor_retention(scaled, projected, k=5), 5),
            "distance_bins": distance_bins(before, after),
        },
        "clustering": {
            "method": "Ward hierarchical clustering on scaled composite scores",
            "solutions": clusters,
        },
        "methods": {
            "composites": "Sum existing subscale columns within each of seven TCI dimensions",
            "scaling": "MinMaxScaler applied separately to each of seven composite traits",
            "comparisons": "Upper triangle of the Euclidean distance matrix",
            "associations": "Pearson correlations among the seven composite traits",
            "multiplicity": "Benjamini-Hochberg correction across 21 exploratory tests, q < 0.05",
            "stress": "sqrt(sum((original_distance - map_distance)^2) / sum(map_distance^2))",
            "mds": "2 components, four random starts, seed 42, at most 300 iterations, tolerance 1e-6",
        },
    }


def render_page(results: dict[str, Any]) -> str:
    """Embed the aggregate results into a standalone local/Pages HTML document."""
    template = (SITE / "template.html").read_text(encoding="utf-8")
    data = json.dumps(results, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    overview = {
        "[[N_RECORDS]]": str(results["record_count"]),
        "[[N_SUBSCALES]]": str(results["subscale_count"]),
        "[[N_PAIRS]]": f"{results['profile_pairs']:,}",
        "[[N_SIGNIFICANT]]": str(results["fdr_significant"]),
        "[[SPEARMAN]]": f"{results['projection']['distance_spearman']:.2f}",
        "[[STRESS]]": f"{results['projection']['stress_1']:.3f}",
        "[[RETAINED]]": f"{results['projection']['nearest_five_retained']:.1%}",
        "[[SILHOUETTE_4]]": f"{results['clustering']['solutions']['4']['silhouette']:.3f}",
    }
    for token, replacement in overview.items():
        template = template.replace(token, replacement)
    if "[[" in template or "]]" in template:
        raise ValueError("Unresolved placeholder in research page template")
    return template.replace("__EMBEDDED_RESULTS_JSON__", data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE_WORKBOOK, help="Original scored workbook")
    parser.add_argument("--check", action="store_true", help="Check whether generated site files are current")
    args = parser.parse_args()

    workbook = args.input.expanduser().resolve()
    if not workbook.is_file():
        parser.error(f"Cannot find the baseline workbook: {workbook}")
    source = pd.read_excel(workbook, engine="openpyxl")
    results = analyze(composite_scores(source))
    metrics_json = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    index_html = render_page(results)
    targets = [(SITE / "results.json", metrics_json), (SITE / "index.html", index_html)]

    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, content in targets if not path.exists() or path.read_text(encoding="utf-8") != content]
        if stale:
            parser.error("Outdated generated files: " + ", ".join(stale))
        print("Research summary is current.")
    else:
        SITE.mkdir(parents=True, exist_ok=True)
        for path, content in targets:
            path.write_text(content, encoding="utf-8")
        print("Generated docs/results.json and docs/index.html")

    top = results["strongest_pairs"][0]
    print(f"Baseline profiles: {results['record_count']}")
    print(f"Profile comparisons: {results['profile_pairs']}")
    print(f"Exploratory FDR discoveries: {results['fdr_significant']}/{results['correlation_tests']}")
    print(f"Strongest association: {top['a']}/{top['b']}, r={top['r']:.3f}")
    print(f"MDS distance-rank correlation: {results['projection']['distance_spearman']:.3f}")
    print(f"MDS Stress-1: {results['projection']['stress_1']:.3f}")
    print(f"Four-cluster silhouette: {results['clustering']['solutions']['4']['silhouette']:.3f}")


if __name__ == "__main__":
    main()
