"""Utility script to generate questions.json from fullQuestionnare.txt.

This is a cleaned up version of the original script. It preserves the same
behavior:

- Reads fullQuestionnare.txt from the data directory
- Extracts lines that start with a question number and text
- Skips question 241
- Writes data/questions.json with the structure:
    {
      "questions": [
        {"id": 1, "text": "..."},
        {"id": 2, "text": "..."},
        ...
      ]
    }

Only small improvements:
- Uses pathlib for paths instead of raw '../data/...'
- Adds some basic error checking and comments
"""

import json
import re
from pathlib import Path


def main() -> None:
    # Resolve paths relative to this file:
    #   src/affinitree/convert_questions.py
    #   data/fullQuestionnare.txt
    #   data/questions.json
    base_directory = Path(__file__).resolve().parent
    data_directory = base_directory.parent / "data"

    source_path = data_directory / "fullQuestionnare.txt"
    target_path = data_directory / "questions.json"

    if not source_path.exists():
        raise FileNotFoundError(f"Source questionnaire file not found at {source_path}")

    # Read all lines, stripping trailing newline and spaces
    with source_path.open("r", encoding="utf-8", errors="replace") as file_object:
        lines = [line.strip() for line in file_object]

    questions = []
    index = 0

    # Pattern: leading digits, then at least one space, then text
    number_pattern = re.compile(r"^(\d+)\s+(.*)$")

    while index < len(lines):
        line = lines[index]

        match = number_pattern.match(line)
        if match:
            question_number = int(match.group(1))
            question_text = match.group(2).strip()

            # Preserve original behavior: skip question 241
            if question_number != 241:
                questions.append(
                    {
                        "id": question_number,
                        "text": question_text,
                    }
                )

            # Advance to next blank line (skip any wrapped lines, if present)
            index = index + 1
            while index < len(lines) and lines[index].strip() != "":
                index = index + 1
        else:
            index = index + 1

    # Write questions.json
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as file_object:
        json.dump({"questions": questions}, file_object, indent=2, ensure_ascii=False)

    print(f"Wrote {len(questions)} questions to {target_path}")


if __name__ == "__main__":
    main()