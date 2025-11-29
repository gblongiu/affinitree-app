from __future__ import annotations

"""
Flask application for the Affinitree project.

This application serves:

- The main Affinitree visualization at `/`
- The Temperament and Character Inventory (TCI) questionnaire user interface at `/tci`
- An API endpoint to fetch questionnaire items at `/api/questions`
- An API endpoint to submit questionnaire responses at `/api/submit`

On questionnaire submission, the application:

1. Computes TCI trait scores from the answers
2. Appends a new row to the user score table
3. Optionally rebuilds the Affinitree visualization

To run in development from the project root:

    python web/app.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)

import pandas as pandas_module

# ---------------------------------------------------------------------------
# Path setup so "affinitree" can be imported when running web/app.py directly
# ---------------------------------------------------------------------------

CURRENT_FILE_PATH: Path = Path(__file__).resolve()
PROJECT_ROOT_DIRECTORY: Path = CURRENT_FILE_PATH.parents[1]
SOURCE_DIRECTORY: Path = PROJECT_ROOT_DIRECTORY / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from affinitree.config import AffinitreeConfig  # type: ignore  # noqa: E402
from affinitree.html_builder import build_affinitree_html  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# Flask application and logging
# ---------------------------------------------------------------------------

application = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

AFFINITREE_CONFIGURATION = AffinitreeConfig()

# ---------------------------------------------------------------------------
# Questionnaire loading and scoring
# ---------------------------------------------------------------------------


def load_questionnaire(affinitree_configuration: AffinitreeConfig) -> Dict[str, Any]:
    """
    Load questionnaire metadata from the questions.json file.

    Expected JSON format (example per question):

    {
      "questions": [
        {
          "id": "Q1",
          "text": "I often seek new and exciting experiences.",
          "trait": "NS",
          "reverse_scored": false,
          "scale_min": 1,
          "scale_max": 5
        },
        ...
      ]
    }
    """
    questionnaire_path: Path = affinitree_configuration.questionnaire_json_path
    if not questionnaire_path.exists():
        message = f"Questionnaire JSON not found at {questionnaire_path}"
        logger.error(message)
        raise FileNotFoundError(message)

    with questionnaire_path.open("r", encoding="utf-8") as file_object:
        questionnaire_data: Dict[str, Any] = json.load(file_object)

    if not isinstance(questionnaire_data, dict) or "questions" not in questionnaire_data:
        logger.warning("questions.json does not have a top level 'questions' key")

    return questionnaire_data


def compute_tci_scores(
    answers: Dict[str, Any],
    questionnaire: Dict[str, Any],
    affinitree_configuration: AffinitreeConfig,
) -> Dict[str, float]:
    """
    Compute Temperament and Character Inventory (TCI) trait scores from questionnaire answers.

    Parameters
    ----------
    answers
        Mapping from question identifier to numeric response.
    questionnaire
        Questionnaire metadata loaded from JSON. Must contain a "questions" list.
    affinitree_configuration
        AffinitreeConfig instance used to determine which trait codes to use.

    Returns
    -------
    Dict[str, float]
        Mapping from score column name (for example "NS Total") to total score.
    """
    question_list = questionnaire.get("questions", [])
    if not isinstance(question_list, list):
        logger.error("Questionnaire 'questions' field is not a list")
        raise ValueError("Invalid questionnaire JSON format")

    trait_totals: Dict[str, float] = {
        trait_code: 0.0 for trait_code in affinitree_configuration.distance_trait_codes
    }

    trait_counts: Dict[str, int] = {
        trait_code: 0 for trait_code in affinitree_configuration.distance_trait_codes
    }

    from affinitree.preprocessing import trait_column_name_from_code  # type: ignore

    for question in question_list:
        try:
            question_identifier = str(question["id"])
            trait_code = str(question["trait"])
            reverse_scored = bool(question.get("reverse_scored", False))
            scale_minimum = float(question.get("scale_min", 1))
            scale_maximum = float(question.get("scale_max", 5))
        except KeyError as exception:
            logger.warning("Skipping question with missing key: %s", exception)
            continue

        if trait_code not in trait_totals:
            # Ignore traits that are not part of the current pipeline
            continue

        raw_value = answers.get(question_identifier)
        if raw_value is None:
            continue

        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            logger.warning("Non numeric answer for %s: %r", question_identifier, raw_value)
            continue

        numeric_value = max(scale_minimum, min(scale_maximum, numeric_value))

        if reverse_scored:
            score = scale_maximum + scale_minimum - numeric_value
        else:
            score = numeric_value

        trait_totals[trait_code] += score
        trait_counts[trait_code] += 1

    result: Dict[str, float] = {}
    for trait_code, total_score in trait_totals.items():
        column_name = trait_column_name_from_code(trait_code)
        result[column_name] = float(total_score)

    logger.info(
        "Computed TCI scores for traits: %s",
        ", ".join(sorted(result.keys())),
    )
    return result


def append_user_scores_row(
    scores: Dict[str, float],
    affinitree_configuration: AffinitreeConfig,
) -> None:
    """
    Append a new row of scores to the user data file.

    If the file does not exist, a new table is created with this row.
    """
    user_data_path = affinitree_configuration.user_data_path
    if user_data_path is None:
        message = "No user_data_path configured in AffinitreeConfig"
        logger.error(message)
        raise RuntimeError(message)

    user_data_path = Path(user_data_path)

    new_row_data: Dict[str, Any] = {**scores, "source": "user"}
    new_row_dataframe = pandas_module.DataFrame([new_row_data])

    if user_data_path.exists():
        suffix = user_data_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            existing_dataframe = pandas_module.read_excel(user_data_path)
        elif suffix == ".csv":
            existing_dataframe = pandas_module.read_csv(user_data_path)
        else:
            message = f"Unsupported user data file extension: {suffix}"
            raise ValueError(message)

        combined_dataframe = pandas_module.concat(
            [existing_dataframe, new_row_dataframe],
            ignore_index=True,
        )
    else:
        combined_dataframe = new_row_dataframe

    suffix = user_data_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        combined_dataframe.to_excel(user_data_path, index=False)
    elif suffix == ".csv":
        combined_dataframe.to_csv(user_data_path, index=False)
    else:
        message = f"Unsupported user data file extension: {suffix}"
        raise ValueError(message)

    logger.info("Appended new user scores row to %s", user_data_path)


def rebuild_affinitree_html(affinitree_configuration: AffinitreeConfig) -> None:
    """
    Rebuild the Affinitree index.html file on disk.

    This is useful when you want a static snapshot for hosting.
    """
    html_string = build_affinitree_html(affinitree_configuration)
    output_path = PROJECT_ROOT_DIRECTORY / "index.html"
    output_path.write_text(html_string, encoding="utf-8")
    logger.info("Wrote updated Affinitree HTML to %s", output_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@application.route("/", methods=["GET"])
def index() -> str:
    """
    Main Affinitree visualization.

    In development, this regenerates the HTML on each request.
    """
    html_string = build_affinitree_html(AFFINITREE_CONFIGURATION)
    return html_string


@application.route("/tci", methods=["GET"])
def tci_page():
    """
    Render the TCI questionnaire user interface template.
    """
    return render_template("tci_test.html")


@application.route("/api/questions", methods=["GET"])
def api_questions():
    """
    Return questionnaire questions as JSON.
    """
    try:
        questionnaire = load_questionnaire(AFFINITREE_CONFIGURATION)
    except FileNotFoundError as exception:
        return jsonify({"error": str(exception)}), 500

    return jsonify(questionnaire)


@application.route("/api/submit", methods=["POST"])
def api_submit():
    """
    Accept questionnaire responses and update user scores.

    Expected JSON payload:

    {
      "answers": {
        "Q1": 4,
        "Q2": 2,
        ...
      },
      "metadata": {
        "name": "Optional display name",
        "email": "optional@example.com"
      }
    }
    """
    payload: Dict[str, Any] | None = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON"}), 400

    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return jsonify({"error": "'answers' must be an object mapping id to value"}), 400

    try:
        questionnaire = load_questionnaire(AFFINITREE_CONFIGURATION)
        scores = compute_tci_scores(answers, questionnaire, AFFINITREE_CONFIGURATION)
        append_user_scores_row(scores, AFFINITREE_CONFIGURATION)
        rebuild_affinitree_html(AFFINITREE_CONFIGURATION)
    except Exception as exception:
        logger.exception("Failed to process questionnaire submission")
        return jsonify({"error": str(exception)}), 500

    return jsonify({"status": "ok", "scores": scores})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # For local development. In production, a WSGI server should be used instead.
    application.run(host="127.0.0.1", port=5000, debug=True)