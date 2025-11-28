import os
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go

print(os.getcwd())

BASE_DATA_PATH = "Neuma_TCI_Score.xlsx"
USER_DATA_PATH = "Neuma_TCI_Score_user.xlsx"


def load_table(path: str) -> pd.DataFrame:
    """
    Load either an Excel or CSV table, based on file extension.
    Excel files are read with openpyxl; CSV uses pandas' default.
    """
    ext = os.path.splitext(path)[1].lower()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path!r} does not exist in {os.getcwd()!r}")
    if ext in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type for {path!r}")


# Load base data and optional user data, then stack them
df_base = load_table(BASE_DATA_PATH)

if os.path.exists(USER_DATA_PATH):
    df_user = load_table(USER_DATA_PATH)
    df = pd.concat([df_base, df_user], ignore_index=True)
else:
    df = df_base.copy()

# --- Compute *Total columns in Python so we are not limited by Excel formulas ---

TRAIT_GROUPS = {
    # total column      # contributing item columns
    "NS Total": ["NS1", "NS2", "NS3", "NS4"],
    "HA Total": ["HA1", "HA2", "HA3", "HA4"],
    "RD Total": ["RD1", "RD2", "RD3", "RD4"],
    "P Total":  ["P1",  "P2",  "P3",  "P4"],
    "SD Total": ["SD1", "SD2", "SD3", "SD4", "SD5"],
    "CO Total": ["CO1", "CO2", "CO3", "CO4", "CO5"],
    "ST Total": ["ST1", "ST2", "ST3"],
}

# Make sure all contributing columns are numeric and compute totals as row-wise means
for total_col, cols in TRAIT_GROUPS.items():
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    # This mirrors Excel's =AVERAGE(): mean across available items, NaN if all are NaN
    df[total_col] = df[cols].mean(axis=1)


# --- Helper functions for visualization ---


def assign_color(row_data: pd.Series) -> str:
    """
    Assign a node color based on which temperament trait is dominant.
    Uses HA, NS, RD totals (already computed in Python).
    """
    ha = row_data.get("HA Total", np.nan)
    ns = row_data.get("NS Total", np.nan)
    rd = row_data.get("RD Total", np.nan)

    # If any required trait is missing, fall back to a neutral color
    if not np.isfinite(ha) or not np.isfinite(ns) or not np.isfinite(rd):
        return "gray"

    dominant_trait = max(ha, ns, rd)

    if dominant_trait == ha:
        return "cyan"
    elif dominant_trait == ns:
        return "yellow"
    else:
        return "magenta"


def create_radial_bar_chart(df: pd.DataFrame, node_index: int) -> str:
    """
    Create a radial bar chart for a single row of df and return it as a base64 PNG string.
    """
    values = df.loc[
        node_index,
        ["P Total", "HA Total", "NS Total", "RD Total", "SD Total", "ST Total", "CO Total"],
    ].values

    user_handle = df.loc[node_index, "Identifier"]

    categories = ["P", "HA", "NS", "RD", "SD", "ST", "CO"]
    n_categories = len(categories)

    rotation_offset = np.pi / 14
    angles = [(n / float(n_categories) * 2 * np.pi) + rotation_offset for n in range(n_categories)]
    angles += angles[:1]

    # Normalize values safely, even if they are all zeros or NaN
    values = values.astype(float)
    max_value = np.nanmax(values)

    if not np.isfinite(max_value) or max_value <= 0:
        # All zeros or invalid -> use a flat dummy ring so we don't get NaNs
        normalized_values = [1.0] * len(values)
    else:
        normalized_values = [
            (float(v) / max_value) * 3 if np.isfinite(v) else 0.0 for v in values
        ]

    max_norm = max(normalized_values) if normalized_values else 1.0

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"polar": True})

    custom_colors = ["#8B4513", "#FF0000", "#FFA500", "#FF69B4", "#0000FF", "#7E2F94", "#4CAF50"]
    custom_colors_legend = ["#8B4513", "#FF0000", "#FFA500", "#FF69B4", "#4CAF50", "#7E2F94", "#0000FF"]

    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in custom_colors_legend]

    for i in range(n_categories):
        ax.bar(angles[i], normalized_values[i], width=0.4, color=custom_colors[i], alpha=0.7)

    plt.xticks(angles[:-1], categories, fontsize=12, fontweight="bold", color="gray")
    ax.set_yticklabels([])

    ax.set_ylim(0, max_norm * 1.1 if max_norm > 0 else 1.0)

    ax.set_facecolor("#F5F5F5")
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.title(f"{user_handle}", fontsize=18, fontweight="bold", color="gray", y=1.1, x=0.7)

    temperament_labels = ["P: Persistence", "HA: Harm Avoidance", "NS: Novelty Seeking", "RD: Reward Dependence"]
    character_labels = ["CO: Cooperativeness", "ST: Self-Transcendence", "SD: Self-Directedness"]

    temperament_legend = ax.legend(
        custom_lines[:4],
        temperament_labels,
        bbox_to_anchor=(1.06, 0.84),
        loc="upper left",
        fontsize=12,
        title="Temperament",
        title_fontsize=14,
    )
    temperament_legend.get_title().set_color("black")
    ax.add_artist(temperament_legend)

    character_legend = ax.legend(
        custom_lines[4:],
        character_labels,
        bbox_to_anchor=(1.09, 0.6),
        loc="upper left",
        fontsize=12,
        title="Character",
        title_fontsize=14,
    )
    character_legend.get_title().set_color("black")

    for text in temperament_legend.texts:
        text.set_color("black")
    for text in character_legend.texts:
        text.set_color("black")

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    print(f"Image for node index {node_index} created with size {len(encoded_image)}")
    return encoded_image


# --- Prepare data for clustering ---

columns_to_normalize = [
    "NS1", "NS2", "NS3", "NS4", "NS Total",
    "HA1", "HA2", "HA3", "HA4", "HA Total",
    "RD1", "RD2", "RD3", "RD4", "RD Total",
    "P1", "P2", "P3", "P4", "P Total",
    "SD1", "SD2", "SD3", "SD4", "SD5", "SD Total",
    "CO1", "CO2", "CO3", "CO4", "CO5", "CO Total",
    "ST1", "ST2", "ST3", "ST Total",
]

# Ensure the columns are numeric, coercing bad strings (like '#DIV/0!') to NaN
df[columns_to_normalize] = df[columns_to_normalize].apply(pd.to_numeric, errors="coerce")

# Drop rows that have any NaNs in the clustering features
before = len(df)
df = df.dropna(subset=columns_to_normalize, how="any").reset_index(drop=True)
after = len(df)
print(f"Dropped {before - after} rows with missing or invalid values in score columns")

# Now scale only the cleaned rows
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df[columns_to_normalize]),
    columns=columns_to_normalize,
)

# --- Clustering ---

cluster = AgglomerativeClustering(n_clusters=4, metric="euclidean", linkage="ward")
df["Role"] = cluster.fit_predict(df_normalized)

# Map numeric labels to semantic roles
role_mapping = {0: "Root", 1: "Trunk", 2: "Branch", 3: "Leaf"}
df["Role"] = df["Role"].map(role_mapping)

# Assign colors based on dominant temperament traits
df["Color"] = df.apply(assign_color, axis=1)

# --- Similarity and layout ---

similarity_matrix = pairwise_distances(df_normalized)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
pos = mds.fit_transform(similarity_matrix)

# Positions + attributes frame
mds_df = pd.DataFrame(pos, columns=["x", "y"])
mds_df["Role"] = df["Role"]
mds_df["Color"] = df["Color"]

# --- Build graph ---

G = nx.Graph()

for index, current_row in mds_df.iterrows():
    identifier = df.loc[index, "Identifier"]
    G.add_node(
        identifier,
        role=current_row["Role"],
        color=current_row["Color"],
        **{"P Total": df.loc[index, "P Total"]},
    )

# Simple layout mapping from identifier to coordinates
layout = {
    node: (
        mds_df.loc[df[df["Identifier"] == node].index[0], "x"],
        mds_df.loc[df[df["Identifier"] == node].index[0], "y"],
    )
    for node in G.nodes()
}


def get_edge_trace(input_graph: nx.Graph, input_layout: dict) -> go.Scatter:
    edge_x = []
    edge_y = []
    for edge in input_graph.edges():
        x0, y0 = input_layout[edge[0]]
        x1, y1 = input_layout[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="rgba(128, 128, 128, 0.5)", width=1),
    )


def get_node_trace(input_graph: nx.Graph, input_layout: dict) -> go.Scatter:
    """
    Build the node trace AND attach the correct radial chart image
    for each node by matching on Identifier.
    """
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_custom = []

    for node in input_graph.nodes():
        x, y = input_layout[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(input_graph.nodes[node]["color"])
        node_text.append(f"{node} ({input_graph.nodes[node]['role']})")

        # Align customdata explicitly to the same node order using Identifier
        idx_list = df.index[df["Identifier"] == node].tolist()
        if idx_list:
            idx = idx_list[0]
            node_custom.append(create_radial_bar_chart(df, idx))
        else:
            node_custom.append(None)

    return go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        text=node_text,
        hoverinfo="text",
        customdata=node_custom,
        marker=dict(
            showscale=False,
            colorscale="YlGnBu",
            reversescale=True,
            color=node_colors,
            size=10,
            line_width=2,
        ),
    )


edge_trace = get_edge_trace(G, layout)
node_trace = get_node_trace(G, layout)

# Update the hovertemplate
node_trace["hovertemplate"] = "<b>%{text}</b><br><extra></extra>"

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        # font now lives inside the title dict
        title=dict(text="Affinitree", x=0.5, font=dict(size=26)),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        clickmode="event+select",
        autosize=False,
        width=1400,
        height=800,
    ),
)

# Print the number of nodes in each role
role_counts = df["Role"].value_counts()
print(role_counts)

html_string = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Affinitree</title>
    <style>
        #affinitree-plot {{
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 100%;
        }}

        @media (max-width: 767px) {{
            #affinitree-plot {{
                width: 100%;
                height: 100%;
                max-width: none;
                height: auto;
            }}
        }}

        #modal-image {{
            max-width: 100vw;
            max-height: 100vh;
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
        }}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <nav>
        <a href="tci_test.html">Take the TCI questionnaire</a>
    </nav>
    <div id="affinitree-plot"></div>
    <div id="radial-chart-container"></div>
    <script id="plot-data" type="application/json">{fig.to_json()}</script>
    <script src="affinitree.js"></script>
</body>
</html>
"""

# Save the html_string to a file named index.html
with open("index.html", "w") as f:
    f.write(html_string)
