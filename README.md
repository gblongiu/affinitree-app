# Affinitree

Affinitree is an interactive visualization and questionnaire system for Temperament and Character Inventory (TCI) data. It turns raw TCI scores into a two dimensional forest of nodes where each person is a tree. Clicking a node shows a radial profile of temperament and character traits, and the layout lets you see who is shaped most similarly at a glance.

Affinitree is built as a small Python package plus a Flask web app. The core pipeline is:

1. Load base TCI scores and optional user scores  
2. Normalize trait scores and compute pairwise distances  
3. Embed individuals into 2D space with multidimensional scaling (MDS)  
4. Cluster individuals and assign roles such as Root, Trunk, Branch, Leaf  
5. Build an interactive Plotly figure plus radial trait charts  
6. Serve the visualization and questionnaire through a web interface  

---

## Features

- Interactive Plotly graph of all respondents  
- Clickable nodes that show a radial bar chart of temperament and character  
- Hierarchical clustering into four high level roles  
- Simple TCI questionnaire web app backed by Flask  
- Regenerates the visualization when new scores are added  
- Data science pipeline split into reusable modules  

---

## Repository layout

The project is structured as a small package plus a web layer.

    affinitree-app/
      README.md
      requirements.txt

      data/
        Neuma_TCI_Score.xlsx          # base TCI scores
        Neuma_TCI_Score_user.xlsx     # optional user scores
        fullQuestionnare.txt          # raw questionnaire text
        questions.json                # parsed questionnaire

      src/
        affinitree/
          __init__.py
          config.py                   # paths, constants, trait metadata
          data_io.py                  # load and merge score tables
          preprocessing.py            # normalization and feature selection
          similarity.py               # distance matrix construction
          embedding.py                # MDS or other 2D embedding
          graph.py                    # networkx graph and clustering logic
          radial_chart.py             # Matplotlib radial chart generation
          figure.py                   # Plotly figure construction
          html_builder.py             # fig -> full index.html string

      web/
        app.py                        # Flask application
        templates/
          index.html                  # visualization shell
          tci_test.html               # questionnaire UI
        static/
          js/
            affinitree.js             # client side interactions
          css/
            style.css                 # optional styling

      scripts/
        build_affinitree.py           # CLI to regenerate index.html
        convert_questions.py          # fullQuestionnare.txt -> questions.json
---

## Installation

### Prerequisites

- Python 3.10 or newer  
- Git  
- A virtual environment tool such as `venv`  

### Setup

Clone the repository and install dependencies into a virtual environment.

    git clone https://github.com/<your-user>/affinitree-app.git
    cd affinitree-app

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate

    pip install --upgrade pip
    pip install -r requirements.txt

---

## Data inputs

Affinitree expects TCI score tables in the `data/` directory.

- `Neuma_TCI_Score.xlsx`  
  - Base dataset of TCI scores  
  - One row per individual  
  - Columns for trait totals such as `NS Total`, `HA Total`, `RD Total`, `P Total`, `SD Total`, `CO Total`, `ST Total` plus any subscales used for the radial chart  

- `Neuma_TCI_Score_user.xlsx` (optional)  
  - Additional rows for new questionnaire submissions  
  - Same schema as the base file  

The questionnaire system uses:

- `data/fullQuestionnare.txt` as the human editable source  
- `data/questions.json` as the structured representation consumed by the web app  

You can regenerate `questions.json` after editing the text file.

    python scripts/convert_questions.py

---

## How the math works

The Affinitree pipeline uses standard tools from the scientific Python stack.

### Normalization

Trait columns are normalized so that each dimension contributes fairly to distance calculations. By default Affinitree applies `MinMaxScaler` from scikit learn across selected trait totals.

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(
        scaler.fit_transform(df[trait_columns]),
        columns=trait_columns,
        index=df.index,
    )

You can change the set of traits or the scaling strategy in `preprocessing.py` and `config.py`.

### Distance calculation

Pairwise distances between individuals are computed with `pairwise_distances` from `sklearn.metrics`. The default metric is Euclidean distance on normalized trait totals.

    from sklearn.metrics import pairwise_distances

    distance_matrix = pairwise_distances(df_norm.values, metric="euclidean")

The distance metric and any weights are configurable. This makes it possible to experiment with correlation or cosine distance if you want to focus on pattern rather than absolute level.

### Dimensionality reduction

To visualize the high dimensional trait space the project uses multidimensional scaling to embed individuals into two dimensions while trying to preserve pairwise distances.

    from sklearn.manifold import MDS

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
    )
    coords = mds.fit_transform(distance_matrix)

The resulting coordinates are used as node positions in the graph. Other embedding methods can be added in `embedding.py`.

### Clustering and roles

Hierarchical clustering groups individuals into a small number of clusters which are labeled with high level roles such as Root, Trunk, Branch, Leaf. This is implemented in `graph.py` with `AgglomerativeClustering` from scikit learn and then stored as node metadata on the networkx graph.

---

## Radial trait chart

Each node has a radial bar chart that summarizes temperament and character traits.

Temperament:

- P  Persistence  
- HA Harm Avoidance  
- NS Novelty Seeking  
- RD Reward Dependence  

Character:

- CO Cooperativeness  
- ST Self Transcendence  
- SD Self Directedness  

`radial_chart.py` converts one row of trait totals into a Matplotlib polar plot and returns a base64 encoded PNG that is attached to the Plotly node as custom data. When you click a node the front end displays this chart next to the graph.

---

## Running the Affinitree visualization

You can regenerate the static visualization from the command line. This reads the data files, rebuilds the entire graph, and writes `index.html` to the project root.

    source .venv/bin/activate
    python scripts/build_affinitree.py

After the script completes, open `index.html` in a browser and interact with the graph. Click on any node to see its radial trait chart.

If you host the repository through GitHub Pages, commit and push the new `index.html` to update the live demo.

---

## Running the TCI questionnaire web app

The Flask app serves the questionnaire and can append new results to the user score table, then trigger a rebuild of the visualization.

1. Make sure dependencies are installed and the virtual environment is active.  
2. Start the server.

       python web/app.py

3. Open the questionnaire in your browser.

       http://127.0.0.1:5000/tci

4. Complete the TCI questionnaire and submit answers.  
5. The backend scores the responses, writes them into `Neuma_TCI_Score_user.xlsx` or another configured store, and may call the Affinitree pipeline to regenerate the graph.

The exact scoring logic lives in `web/app.py` and uses `data/questions.json` as the source of truth for question metadata.

---

## Configuration

Several aspects of Affinitree can be adjusted without editing the entire pipeline.

Look in `src/affinitree/config.py` for:

- Paths to base and user score files  
- The list of trait columns used for distance and for the radial chart  
- Number of clusters and their role labels  
- Default distance metric and embedding method  

You can also parameterize the pipeline further by passing a configuration object into the functions in `data_io.py`, `preprocessing.py`, and `embedding.py`.

---

## Development

To work on the project itself:

1. Install dependencies.

       pip install -r requirements.txt

2. Run the pipeline on a small sample to confirm your environment.

       python scripts/build_affinitree.py

3. Start the Flask app during development.

       export FLASK_APP=web.app
       export FLASK_ENV=development
       flask run

4. Consider adding tests under `tests/` and wiring up continuous integration through GitHub Actions so that the pipeline and questionnaire scoring are exercised on every push.

---

## Acknowledgements

Affinitree is a creation by Affinitree1 and friends. The project stands on the shoulders of many open source libraries, especially NumPy, pandas, scikit learn, networkx, Matplotlib, Plotly, and Flask.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.