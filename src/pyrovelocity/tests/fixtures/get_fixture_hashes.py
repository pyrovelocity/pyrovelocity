"""
Execute this script to obtain SHA-256 hashes of fixture files.

python src/pyrovelocity/tests/fixtures/get_fixture_hashes.py

This will print the hashes of all fixture files, which can be used in conftest.py
"""

from importlib.resources import files

from pyrovelocity.io.hash import hash_file
from pyrovelocity.logging import configure_logging

logger = configure_logging(__name__)


def get_fixture_hashes():
    """Get SHA-256 hashes of all fixture files."""

    data_dir = files("pyrovelocity.tests.data")

    fixture_files = [
        "pancreas_50_13.json",
        "preprocessed_pancreas_50_7.json",
        "trained_pancreas_50_7.json",
        "postprocessed_pancreas_50_7.json",
        "larry_multilineage_50_6.json",
        "preprocessed_larry_multilineage_50_6.json",
        "trained_larry_multilineage_50_6.json",
        "postprocessed_larry_multilineage_50_6.json",
    ]

    print("\nAdd these expected hashes to conftest.py:\n")
    print("FIXTURE_HASHES = {")

    for fixture_file in fixture_files:
        file_path = data_dir / fixture_file
        if file_path.exists():
            file_hash = hash_file(file_path)
            print(f'    "{fixture_file}": "{file_hash}",')
        else:
            print(f'    # "{fixture_file}": "<file not found>",')

    print("}\n")


if __name__ == "__main__":
    get_fixture_hashes()
