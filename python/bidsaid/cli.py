"""Helpers for building command-line interfaces with ``argparse``."""

import argparse
from typing import Literal
from typing import Any, Callable


def none_or(cast: Callable[[str], Any] = str) -> Callable[[str], Any | None]:
    """
    Create an ``argparse`` Type That Accepts "None".

    Returns a converter that maps the string "none" (case-insensitive) to ``None`` and
    passes every other value through ``cast``. Useful for optional arguments that have a
    non-None default but should be disableable from the command line.

    Parameters
    ----------
    cast : :obj:`Callable[[str], Any]`, default=str
        The function used to convert values other than "none" (e.g. ``float`` or ``int``).

    Returns
    -------
    Callable[[str], Any | None]
        A converter suitable for the ``type`` argument of ``ArgumentParser.add_argument``.

    Examples
    --------
    >>> import argparse
    >>> from bidsaid.cli import none_or
    >>> parser = argparse.ArgumentParser()
    >>> _ = parser.add_argument("--fwhm", type=none_or(float), default=6.0)
    >>> parser.parse_args(["--fwhm", "none"]).fwhm is None
    True
    >>> parser.parse_args(["--fwhm", "4"]).fwhm
    4.0
    """

    def _convert(value: str) -> Any | None:
        if str(value).strip().lower() == "none":
            return None

        return cast(value)

    return _convert


def str_to_bool(value: str):
    """
    Create an ``argparse`` Type That Creates a boolean.

    Converts case-insensitive strings to true or false/

    Parameters
    ----------
    value : :obj:`str`
        String to convert to boolean. Valid inputs are "t", "true", "1", "y", "yes", "f", "false", "0", "n", and "no".

    Returns
    -------
    bool
    """
    value = str(value)
    if value.lower() not in [
        "t",
        "true",
        "1",
        "y",
        "yes",
        "f",
        "false",
        "0",
        "n",
        "no",
    ]:
        raise ValueError(
            "valid inputs are 't', 'true', '1', 'y', 'yes', 'f', 'false', '0', 'n', 'no'"
        )

    return True if value.lower() in ["t", "true", "1", "y", "yes"] else False


def n_dummy_type(value: Literal["auto"] | int):
    value = str(value)

    if value == "auto":
        return "auto"
    else:
        try:
            value = int(value)

            return value
        except ValueError:
            raise argparse.ArgumentTypeError(
                "`n_dummy_scans` must be 'auto' or an integer."
            )


__all__ = ["none_or", "str_to_bool", "n_dummy_type"]
