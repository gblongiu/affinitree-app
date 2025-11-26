## Affinitree Visualization Project

Table of Contents

1. [Affinitree](#affinitree)
2. [Mathematical Operations](#mathematical-operations)
   - [Normalization](#normalization)
   - [Distance Calculation](#distance-calculation)
   - [Dimensionality Reduction](#dimensionality-reduction)
3. [TCI Questionnaire Web App](#tci-questionnaire-web-app)
4. [How to Run](#how-to-run)
5. [Acknowledgements](#acknowledgements)
6. [License](#license)

### Affinitree

Affinitree is an interactive data visualization project that uses the Plotly Python graphing library. It utilizes a hierarchical clustering algorithm to categorize data and constructs a graph of interconnected nodes from the results. Each node represents an individual with certain character traits. By clicking on a node, a radial bar chart, which displays the individual's character traits, pops up.

The core visualization consists of these source files:

- `affinitreeBeta.py`: The primary Python script performing data preprocessing, clustering, graph creation, and radial bar chart generation.
- `affinitree.js`: A JavaScript file that enhances the interactivity of the Plotly graph.
- `Neuma_TCI_Score.csv`: The dataset containing TCI personality scores used for clustering and visualization.

Running affinitreeBeta.py generates an index.html file that showcases the interactive graph. In this repository, index.html is checked in so that GitHub Pages can serve a live demo, but you can still regenerate it locally by running the script (which will overwrite the existing file).

### Mathematical Operations

The Affinitree visualization tool computes distances between individuals based on their TCI personality test scores. The process includes the following steps:

#### Normalization

The raw TCI scores are normalized using the `MinMaxScaler` from the `sklearn` library. This scaling transforms each score to a range between 0 and 1, which ensures that each dimension contributes equally to the distance calculation.

    from sklearn.preprocessing import MinMaxScaler

    columns_to_normalize = [...]  # List of columns to normalize
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df[columns_to_normalize]),
        columns=columns_to_normalize
    )

#### Distance Calculation

Pairwise distances are computed between every pair of individuals using the Euclidean distance metric. The `pairwise_distances` function from `sklearn.metrics` is used for this computation. The Euclidean distance formula is:

distance(P, Q) = sqrt((x1 - y1)^2 + (x2 - y2)^2 + ... + (xn - yn)^2)

where P and Q are the normalized TCI score vectors for two individuals.

    from sklearn.metrics import pairwise_distances

    similarity_matrix = pairwise_distances(df_normalized)

#### Dimensionality Reduction

Multidimensional scaling (MDS) is used to reduce the dimensionality of the data from the original number of TCI dimensions to two dimensions. This is achieved using the `MDS` function from the `sklearn.manifold` library. MDS aims to preserve the pairwise distances between the data points while reducing the dimensions, allowing for a meaningful visualization in 2D space.

    from sklearn.manifold import MDS

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    pos = mds.fit_transform(similarity_matrix)

With these mathematical operations, the Affinitree visualization tool calculates distances using the Euclidean distance formula and generates a 2D representation of the hierarchical data based on individuals' TCI personality test scores.

---

## TCI Questionnaire Web App

In addition to the visualization, this repo includes a simple web app for administering and scoring a multiple choice TCI questionnaire.

Key files:

- `app.py`: A Flask application that serves the questionnaire UI, exposes endpoints to load questions, accepts submitted answers, and computes scores.
- `tci_test.html`: The main HTML page for the questionnaire. It loads questions dynamically and posts responses back to the backend.
- `questions.json`: A structured JSON file containing the TCI questions and options used by the web app.
- `fullQuestionnare.txt`: The original raw questionnaire text used as the source for `questions.json`.
- `convert_questions.py`: A helper script that converts `fullQuestionnare.txt` into `questions.json`, making it easy to regenerate or update the questionnaire.

Typical flow:

1. Maintain or edit the questionnaire in `fullQuestionnare.txt`.
2. Run `convert_questions.py` to regenerate `questions.json`.
3. Start the Flask app (`app.py`) and open the questionnaire page in a browser.
4. Answer the questions and submit to view scores or feedback.

---

## How to Run

The project runs in Python 3.9+ and uses the following libraries:

- pandas  
- numpy  
- networkx  
- matplotlib  
- scikit-learn  
- plotly  
- flask  

Standard library modules such as `base64` and `os` are also used but do not require installation.

### Install dependencies

If you have not installed the required libraries yet, you can do so via `pip`:

    pip install pandas numpy networkx matplotlib scikit-learn plotly flask

### Run the Affinitree visualization

    python affinitreeBeta.py

The script reads the data from Neuma_TCI_Score.csv, performs hierarchical clustering, and generates an interactive graph visualization of the clustered data. The resulting Plotly figure is saved as index.html in the current directory (the same file that is committed in this repo and used by GitHub Pages).

Open `index.html` in a web browser to view and interact with the graph. Click on a node to view a radial bar chart of the individual's personality traits.

### Run the TCI questionnaire app

    python app.py

By default, the Flask server will listen on `http://127.0.0.1:5000`. Open that URL in a web browser to access the questionnaire UI.

1. Load the questionnaire page (served by `tci_test.html`).
2. Answer all questions shown.
3. Submit your responses to have them scored by the backend.
4. Review the returned scores or results.

If you modify `fullQuestionnare.txt`, rerun `convert_questions.py` before restarting the app so that `questions.json` stays in sync.

The code is documented and designed to be customizable. If you encounter issues, confirm that your Python environment has all necessary packages installed and that you are running the commands from the project directory.

---

## Acknowledgements

This project is a creation by **Affinitree1 & Friends**.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
