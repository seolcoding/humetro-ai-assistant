import os
from pathlib import Path


def get_root():
    current = Path(__file__)
    threshold = 5
    while True:
        if "pyproject.toml" in os.listdir(current):
            return current
        current = current.parent
        threshold -= 1
        if threshold == 0:
            raise FileNotFoundError
