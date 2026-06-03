"""Helper functions."""

from pathlib import Path
from typing import Any


def iterable_to_str(str_list: list[str]) -> None:
    """Converts an iterable containing strings to strings."""
    return ", ".join(["'{a}'".format(a=x) for x in str_list])


def is_path(object: Any) -> bool:
    "Determine if input is a Path or string."
    return isinstance(object, (str, Path))
