"""Data validation tests for data/iris.csv.

Run with:  python src/validate.py
      or:  python -m unittest src.validate

Dependency-free: uses only the standard library (csv + unittest), so it works
in the project venv even though pandas/pytest are not installed.
"""

import csv
import os
import unittest

# Resolve repo root from this file's location (src/validate.py -> repo root).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "iris.csv")

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABEL_COLUMN = "species"
EXPECTED_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]
EXPECTED_SPECIES = {"setosa", "versicolor", "virginica"}

# Plausible physical ranges for iris measurements (cm). Wider than the observed
# data on purpose: catches corrupt/garbage values without over-fitting.
MIN_CM = 0.0
MAX_CM = 10.0


def load_rows():
    """Return (header, list_of_row_dicts) read from the CSV."""
    with open(DATA_PATH, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = [row for row in reader]
    return header, rows


class IrisDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header, cls.rows = load_rows()

    def test_file_exists_and_not_empty(self):
        self.assertTrue(os.path.isfile(DATA_PATH), f"missing {DATA_PATH}")
        self.assertGreater(len(self.rows), 0, "no data rows found")

    def test_expected_schema(self):
        self.assertEqual(
            self.header,
            EXPECTED_COLUMNS,
            f"unexpected columns: {self.header}",
        )

    def test_no_missing_values(self):
        for i, row in enumerate(self.rows, start=2):  # line 2 = first data row
            for column in EXPECTED_COLUMNS:
                value = row.get(column)
                self.assertIsNotNone(value, f"line {i}: missing column {column}")
                self.assertNotEqual(
                    value.strip(),
                    "",
                    f"line {i}: empty value in column {column}",
                )

    def test_feature_types_are_floats(self):
        for i, row in enumerate(self.rows, start=2):
            for column in FEATURE_COLUMNS:
                try:
                    float(row[column])
                except (TypeError, ValueError):
                    self.fail(
                        f"line {i}: {column}={row[column]!r} is not numeric"
                    )

    def test_feature_ranges_are_reasonable(self):
        for i, row in enumerate(self.rows, start=2):
            for column in FEATURE_COLUMNS:
                value = float(row[column])
                self.assertGreater(
                    value,
                    MIN_CM,
                    f"line {i}: {column}={value} not > {MIN_CM}",
                )
                self.assertLessEqual(
                    value,
                    MAX_CM,
                    f"line {i}: {column}={value} exceeds {MAX_CM}",
                )

    def test_label_values_are_valid(self):
        for i, row in enumerate(self.rows, start=2):
            self.assertIn(
                row[LABEL_COLUMN],
                EXPECTED_SPECIES,
                f"line {i}: unexpected species {row[LABEL_COLUMN]!r}",
            )

    def test_all_classes_present(self):
        seen = {row[LABEL_COLUMN] for row in self.rows}
        self.assertEqual(
            seen,
            EXPECTED_SPECIES,
            f"missing classes: {EXPECTED_SPECIES - seen}",
        )

    def test_no_unexpected_null_tokens(self):
        # Common ways missing values sneak in as strings.
        null_tokens = {"", "na", "nan", "null", "none", "?"}
        for i, row in enumerate(self.rows, start=2):
            for column in EXPECTED_COLUMNS:
                self.assertNotIn(
                    row[column].strip().lower(),
                    null_tokens,
                    f"line {i}: null-like value in column {column}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
