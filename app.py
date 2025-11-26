from flask import Flask, send_from_directory, request, jsonify
import csv
import os
import subprocess
import pandas as pd

app = Flask(__name__, static_url_path='', static_folder='.')

BASE_SCORE_FILE = "Neuma_TCI_Score.csv"
USER_SCORE_FILE = "Neuma_TCI_Score_user.csv"
RESPONSES_FILE = "responses.csv"


def compute_tci_scores(user_id, answers):
    """
    Turn a list of raw answers into a row that matches Neuma_TCI_Score.csv.

    Right now this is a placeholder so that the pipeline works.
    You need to replace the body of this function with the real scoring
    based on your TCI key.
    """
    base_df = pd.read_csv(BASE_SCORE_FILE)
    columns = list(base_df.columns)

    if columns[0] != "Identifier":
        raise RuntimeError("Expected first column in Neuma_TCI_Score.csv to be 'Identifier'")

    row = {"Identifier": user_id}
    for col in columns[1:]:
        # TODO: replace 0.0 with real computed scores for each trait
        row[col] = 0.0

    return row, columns


def append_user_scores(user_id, answers):
    """Append a new scored row to Neuma_TCI_Score_user.csv."""
    row, columns = compute_tci_scores(user_id, answers)
    file_exists = os.path.exists(USER_SCORE_FILE)

    with open(USER_SCORE_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def regenerate_affinitree():
    """
    Run affinitreeBeta.py to rebuild index.html
    using the combined base and user score data.
    """
    script_path = os.path.join(os.path.dirname(__file__), "affinitreeBeta.py")
    subprocess.run(["python3", script_path], check=True)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/tci_test")
def tci_test_page():
    return send_from_directory(".", "tci_test.html")


@app.route("/submit_test", methods=["POST"])
def submit_test():
    data = request.get_json(force=True)
    user_id = data.get("userId", "").strip()
    answers = data.get("answers")

    if not user_id or not isinstance(answers, list):
        return jsonify({"status": "error", "message": "Invalid data"}), 400

    # 1. Save raw answers for later analysis
    resp_exists = os.path.exists(RESPONSES_FILE)
    with open(RESPONSES_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not resp_exists:
            header = ["userId"] + [f"q{i}" for i in range(1, len(answers) + 1)]
            writer.writerow(header)
        writer.writerow([user_id] + answers)

    # 2. Append scored traits for the visualization
    try:
        append_user_scores(user_id, answers)
    except Exception as e:
        # Log the error but still return OK for the questionnaire
        print("Error scoring TCI answers:", e)

    # 3. Regenerate index.html so the graph picks up the new user
    try:
        regenerate_affinitree()
    except Exception as e:
        print("Error regenerating Affinitree index:", e)

    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(debug=True)
