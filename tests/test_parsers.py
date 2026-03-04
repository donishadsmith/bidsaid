from pathlib import Path

import pytest

from nifti2bids.parsers import (
    load_presentation_log,
    load_eprime_log,
    get_presentation_log_date,
)

from ._constants import PRESENTATION_COLUMNS, EPRIME_DATA_NO_CUE


@pytest.fixture(autouse=False, scope="function")
def create_presentation_logfile(tmp_dir):
    sample_log = [
        "Scenario - Flanker task using SPM event_run8.txt file.\n",
        "Logfile written - 09/24/2025 11:40:20\n",
        "\n",
        "Trial\tEvent Type\tCode\tTime\tTTime\tUncertainty\tDuration\tUncertainty\tReqTime\tReqDur\tStim Type\tPair Index\n",
        "\n",
        "1\tPicture\tcrosshairF\t151281\t151235\t1\t78633\t1\t0\t79789\tother\t1\n",
        "1.0\tPort Input\t54.0\t1.51495\t1.51450\t2.0\n",
        "NaN\tNaN\tNaN",
        "" "\n",
        "Event Type\tCode\tTime\tTTime\tUncertainty\tDuration\tUncertainty\tReqTime\tReqDur\tStim Type\tPair Index\n",
        "Some random text",
    ]
    dst_path = Path(tmp_dir.name) / "sample_log.txt"
    with open(dst_path, "w") as f:
        for line in sample_log:
            f.writelines(line)

    yield dst_path

    dst_path.unlink()


@pytest.fixture(autouse=False, scope="function")
def create_eprime_logfile(tmp_dir):
    dst_path = Path(tmp_dir.name) / "sample_log.txt"
    with open(dst_path, "w") as f:
        for line in EPRIME_DATA_NO_CUE:
            f.writelines("\t".join(line) + "\n")

    yield dst_path

    dst_path.unlink()


def test_load_presentation_log(create_presentation_logfile):
    """Test for ``load_presentation_log`` function."""
    import math

    src_file = create_presentation_logfile
    df = load_presentation_log(src_file)
    assert len(df) == 3
    assert all(col in PRESENTATION_COLUMNS for col in df.columns)
    assert df["Event Type"].values.tolist()[:-1] == ["Picture", "Port Input"]
    assert math.isnan(df["Event Type"].values.tolist()[-1])

    assert df.loc[0, "Time"] == 151281

    df = load_presentation_log(src_file, convert_to_seconds=["Time"])
    assert df.loc[0, "Time"] == 15.1281


def test_load_eprime_log(create_eprime_logfile):
    """Test for ``load_eprime_log`` function."""
    src_file = create_eprime_logfile
    df = load_eprime_log(src_file)
    assert len(df) == 6
    assert df.loc[0, "Data.OnsetTime"] == 10000

    df = load_eprime_log(src_file, convert_to_seconds=["Data.OnsetTime"])
    assert df.loc[0, "Data.OnsetTime"] == 10.0


def test_get_presentation_log_date(create_presentation_logfile):
    """Test for ``get_presentation_log_date`` function."""
    src_file = create_presentation_logfile
    date, time = get_presentation_log_date(src_file)
    assert date == "09/24/2025"
    assert time == "11:40:20"
