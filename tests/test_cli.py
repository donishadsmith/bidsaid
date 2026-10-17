import argparse

import pytest

from bidsaid.cli import none_or


@pytest.mark.parametrize("none_string", ["none", "None", "NONE", " none "])
def test_none_or_returns_none(none_string):
    "Test ``none_or`` maps any casing of 'none' to None."
    assert none_or(float)(none_string) is None


@pytest.mark.parametrize(
    "cast, value, expected", [(float, "4", 4.0), (int, "3", 3), (str, "abc", "abc")]
)
def test_none_or_casts_other_values(cast, value, expected):
    "Test ``none_or`` passes other values through the cast."
    result = none_or(cast)(value)

    assert result == expected and isinstance(result, cast)


def test_none_or_with_argparse():
    "Test ``none_or`` as an ``argparse`` type with a non-None default."
    parser = argparse.ArgumentParser()
    parser.add_argument("--fwhm", type=none_or(float), default=6.0)

    assert parser.parse_args([]).fwhm == 6.0
    assert parser.parse_args(["--fwhm", "none"]).fwhm is None
    assert parser.parse_args(["--fwhm", "4"]).fwhm == 4.0
