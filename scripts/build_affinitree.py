from __future__ import annotations

"""
Command line interface to regenerate the Affinitree index.html file.

This script runs the full Affinitree pipeline (data loading, preprocessing,
distance computation, embedding, graph construction, Plotly figure build)
and writes a standalone HTML file containing the visualization.

Expected project layout:

  project_root/
    build_affinitree.py
    src/
      affinitree/
        __init__.py
        config.py
        data_io.py
        preprocessing.py
        similarity.py
        embedding.py
        graph.py
        radial_chart.py
        figure.py
        html_builder.py
    web/
      app.py
      static/
        js/
          affinitree.js
      templates/
        tci_test.html
    data/
      Neuma_TCI_Score.(csv|xlsx)
      user_scores.(csv|xlsx)
      questions.json

Usage examples (from project_root):

  python build_affinitree.py
  python build_affinitree.py --output web/index.html
  python build_affinitree.py --log-level DEBUG
  python build_affinitree.py --dry-run

In dry run mode, the generated HTML is written to standard output instead
of a file.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup so "affinitree" can be imported when running this file directly
# ---------------------------------------------------------------------------

CURRENT_FILE_PATH: Path = Path(__file__).resolve()
PROJECT_ROOT_DIRECTORY: Path = CURRENT_FILE_PATH.parent
SOURCE_DIRECTORY: Path = PROJECT_ROOT_DIRECTORY / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
  sys.path.insert(0, str(SOURCE_DIRECTORY))

from affinitree.config import AffinitreeConfig  # type: ignore  # noqa: E402
from affinitree.html_builder import build_affinitree_html  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_command_line_arguments() -> argparse.Namespace:
  """
  Parse command line arguments for the build_affinitree script.

  Returns
  -------
  argparse.Namespace
      An object with attributes:
        - output_path: str, where to write the generated HTML file
        - log_level: str, logging level name
        - dry_run: bool, whether to write HTML to stdout instead of a file
  """
  argument_parser = argparse.ArgumentParser(
      description="Regenerate the Affinitree index.html visualization file."
  )

  default_output_path = PROJECT_ROOT_DIRECTORY / "index.html"

  argument_parser.add_argument(
      "-o",
      "--output",
      dest="output_path",
      default=str(default_output_path),
      help=(
          "Path to write the generated HTML file. "
          f"Defaults to {default_output_path}"
      ),
  )

  argument_parser.add_argument(
      "--log-level",
      dest="log_level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
      help="Logging verbosity. Defaults to INFO.",
  )

  argument_parser.add_argument(
      "--dry-run",
      dest="dry_run",
      action="store_true",
      help="If provided, write the generated HTML to standard output instead of a file.",
  )

  return argument_parser.parse_args()


# ---------------------------------------------------------------------------
# Main build routine
# ---------------------------------------------------------------------------


def configure_logging(log_level_name: str) -> None:
  """
  Configure global logging for the script.

  Parameters
  ----------
  log_level_name
      One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
  """
  log_level = getattr(logging, log_level_name.upper(), logging.INFO)
  logging.basicConfig(
      level=log_level,
      format="[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
  )


def build_html_document() -> str:
  """
  Run the Affinitree pipeline and return the generated HTML document string.

  This function creates a default AffinitreeConfig instance and passes it
  to build_affinitree_html.
  """
  logger = logging.getLogger(__name__)
  affinitree_configuration = AffinitreeConfig()

  logger.info("Using Affinitree configuration: %s", affinitree_configuration)

  html_string = build_affinitree_html(affinitree_configuration)

  logger.info("Affinitree HTML document generated (length %d characters)", len(html_string))

  return html_string


def write_output(html_string: str, output_path_string: str, dry_run: bool) -> None:
  """
  Write the generated HTML to the requested destination.

  Parameters
  ----------
  html_string
      The complete HTML document string.
  output_path_string
      Path where the HTML file should be written if dry_run is False.
  dry_run
      If True, html_string is written to standard output instead of a file.
  """
  logger = logging.getLogger(__name__)

  if dry_run:
    logger.info("Dry run enabled; writing HTML to standard output.")
    sys.stdout.write(html_string)
    if not html_string.endswith("\n"):
      sys.stdout.write("\n")
    return

  output_path = Path(output_path_string).resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)

  output_path.write_text(html_string, encoding="utf-8")

  logger.info("Wrote Affinitree HTML to %s", output_path)


def main() -> int:
  """
  Entrypoint for the build_affinitree command line script.

  Returns
  -------
  int
      Process exit code. Zero indicates success.
  """
  arguments = parse_command_line_arguments()
  configure_logging(arguments.log_level)

  logger = logging.getLogger(__name__)
  logger.debug("Command line arguments: %s", arguments)

  try:
    html_string = build_html_document()
    write_output(
        html_string=html_string,
        output_path_string=arguments.output_path,
        dry_run=arguments.dry_run,
    )
  except Exception as exception:
    logger.exception("Failed to build Affinitree HTML document: %s", exception)
    return 1

  return 0


if __name__ == "__main__":
  sys.exit(main())