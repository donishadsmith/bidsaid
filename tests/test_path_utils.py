from pathlib import Path

from bidsaid.path_utils import (
    is_valid_date,
    parse_date_from_path,
    get_file_creation_date,
    get_file_acquisition_order,
    sort_by_acquisition_order,
)


def test_is_valid_date():
    """Test for ``is_valid_date``."""
    assert is_valid_date("241010", "%y%m%d")


def test_parse_date_from_path():
    """Test for ``parse_date_from_path``."""
    date = parse_date_from_path("101_240820_mprage_32chan.nii", "%y%m%d")
    assert date == "240820"

    date = parse_date_from_path("101_mprage_32chan.nii", "%y%m%d")
    assert not date

    date = parse_date_from_path(r"Users/users/Documents/101_240820", "%y%m%d")
    assert date == "240820"


def test_get_file_creation_date(tmp_dir):
    """Test for ``get_file_creation_date``."""
    from pathlib import Path

    path = Path(tmp_dir.name) / "test.txt"
    with open(path, "w") as f:
        pass

    date = get_file_creation_date(path, "%Y-%m-%d")
    assert is_valid_date(date, "%Y-%m-%d")


def test_get_file_acquisition_order():
    """Test for ``get_file_acquisition_order``."""
    assert get_file_acquisition_order("101_10_1.nii") == "10_1"
    assert get_file_acquisition_order("101_10.nii") == "10"
    assert not get_file_acquisition_order("101.nii")


def test_sort_by_acquisition_order():
    """Test for ``sort_by_acquisition_order``."""
    filenames = ["101_10_2.nii", "101_10_1.nii", "101_3_1.nii"]
    sorted_filenames = ["101_3_1.nii", "101_10_1.nii", "101_10_2.nii"]
    sorted_filenames = [Path(x) for x in sorted_filenames]
    assert sort_by_acquisition_order(filenames) == sorted_filenames

    filenames = [
        "101_10_2.nii",
        "101_10_1.nii",
        "101_3_1.nii",
        "102_2_1.nii",
        "102_1_1.nii",
    ]
    sorted_filenames = [
        "102_1_1.nii",
        "102_2_1.nii",
        "101_3_1.nii",
        "101_10_1.nii",
        "101_10_2.nii",
    ]
    sorted_filenames = [Path(x) for x in sorted_filenames]
    assert sort_by_acquisition_order(filenames) == sorted_filenames

    filenames = [
        "101_10.nii",
        "101_9.nii",
    ]
    sorted_filenames = [
        "101_9.nii",
        "101_10.nii",
    ]
    sorted_filenames = [Path(x) for x in sorted_filenames]
    assert sort_by_acquisition_order(filenames) == sorted_filenames

    assert sort_by_acquisition_order(filenames, return_filenames_only=False)[0] == (
        1,
        9,
        Path("101_9.nii"),
    )
