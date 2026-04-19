from pathlib import Path

import pandas as pd, pytest

from bidsaid.events import (
    PresentationBlockExtractor,
    PresentationEventExtractor,
    EPrimeBlockExtractor,
    EPrimeEventExtractor,
    add_instruction_timing,
)
from ._constants import (
    BLOCK_PRESENTATION_DATA_NO_CUE,
    BLOCK_PRESENTATION_DATA_WITH_CUE,
    EVENT_PRESENTATION_DATA,
    EPRIME_DATA_NO_CUE,
    EPRIME_DATA_WITH_CUE,
)


def _create_presentation_logfile(dst_dir, data, design):
    dst_dir = Path(dst_dir)

    filename = dst_dir / f"{design}.txt"
    with open(filename, mode="w") as file:
        for line in data:
            file.write("\t".join(line) + "\n")


def test_dataframe_copy():
    """Test to ensure original dataframe is not modified"""
    from bidsaid.events import _process_log_or_df

    df = pd.DataFrame({"A": [1, 2, 3], "B": [5, 6, 7]})
    new_df = _process_log_or_df(
        df,
        convert_to_seconds=["A"],
        initial_column_headers=None,
        divisor=10,
        software=None,
    )

    assert df["A"].values.tolist() == [1, 2, 3]
    assert new_df["A"].values.tolist() == [0.1, 0.2, 0.3]
    assert not id(df) == id(new_df)

    new_df = _process_log_or_df(
        df,
        convert_to_seconds=None,
        initial_column_headers=None,
        divisor=None,
        software=None,
    )

    assert not id(df) == id(new_df)


@pytest.mark.parametrize("scanner_start_time", [None, 10000])
def test_PresentationBlockExtractor(tmp_dir, scanner_start_time):
    """Test for ``PresentationBlockExtractor``."""
    from pandas.testing import assert_frame_equal

    filename = Path(tmp_dir.name) / "block.txt"
    _create_presentation_logfile(tmp_dir.name, BLOCK_PRESENTATION_DATA_NO_CUE, "block")

    expected_df = pd.DataFrame(
        {
            "onset": [0.0, 34.0] if scanner_start_time is None else [1.0, 35.0],
            "duration": [14.0, 14.0],
            "trial_type": ["indoor", "indoor"],
        }
    )

    extractor = PresentationBlockExtractor(
        log_or_df=filename,
        block_cue_names=["indo.*"],
        convert_to_seconds=["Time", "Duration"],
        scanner_event_type="Pulse",
        scanner_trigger_code="99",
        n_discarded_volumes=1 if scanner_start_time is None else 0,
        tr=1 if scanner_start_time is None else None,
        rest_block_codes="res.*",
        quit_code="quit",
    )

    scanner_start_time = scanner_start_time / 10000 if scanner_start_time else None
    onsets = extractor.extract_onsets(scanner_start_time=scanner_start_time)
    durations = extractor.extract_durations()
    trial_types = extractor.extract_trial_types()

    df = pd.DataFrame(
        {"onset": onsets, "duration": durations, "trial_type": trial_types}
    )
    assert_frame_equal(df, expected_df)


def test_PresentationBlockExtractor_mean_rt_and_accuracy(tmp_dir):
    """Test for ``PresentationBlockExtractor.extract_mean_reaction_times`` and ``extract_mean_accuracies``."""
    filename = Path(tmp_dir.name) / "block.txt"
    _create_presentation_logfile(tmp_dir.name, BLOCK_PRESENTATION_DATA_NO_CUE, "block")

    extractor = PresentationBlockExtractor(
        log_or_df=filename,
        block_cue_names=["indoor"],
        convert_to_seconds=["Time"],
        scanner_event_type="Pulse",
        scanner_trigger_code="99",
        rest_block_codes="rest",
        quit_code="quit",
        split_cue_as_instruction=True,
        drop_instruction_cues=True,
    )

    # Test mean RT for all trials (default)
    mean_rts_all = extractor.extract_mean_reaction_times()
    assert len(mean_rts_all) == 2
    # Block 1: Mean = (1.5 + 1.2 + 1.6 + 1.1) / 4 = 1.35s
    assert mean_rts_all[0] == pytest.approx(1.35, rel=1e-2)
    # Block 2: Mean = (1.5 + 1.1 + 1.6 + NaN) / 3 = 1.400s
    assert mean_rts_all[1] == pytest.approx(1.400, rel=1e-2)

    # Test mean RT for correct trials only
    response_map = {"hit": 1, "miss": 0}
    mean_rts_correct = extractor.extract_mean_reaction_times(
        response_map=response_map,
        response_type="correct",
    )
    assert len(mean_rts_correct) == 2
    # Block 1: only correct (1.5 + 1.2 + 1.1) / 3
    assert mean_rts_correct[0] == pytest.approx(1.2667, rel=1e-2)
    # Block 2: Mean = (1.5 + 1.1 + 1.6 + NaN) / 3 = 1.400s
    assert mean_rts_correct[1] == pytest.approx(1.400, rel=1e-2)

    # Test that response_map is required when response_type is not "all"
    with pytest.raises(ValueError, match="response_map"):
        extractor.extract_mean_reaction_times(response_type="correct")

    # Test mean accuracies
    response_map = {"hit": 1, "miss": 0}
    mean_accs = extractor.extract_mean_accuracies(response_map=response_map)
    assert len(mean_accs) == 2
    # Block 1
    assert mean_accs[0] == 0.75
    # Block 2 is missing one response (1 + 1 + 1 + 0) = 0.75
    assert mean_accs[1] == 0.75


@pytest.mark.parametrize("drop_instruction_cues", [True, False])
def test_PresentationBlockExtractor_instruction_cue_separation(
    tmp_dir, drop_instruction_cues
):
    """Test for ``PresentationBlockExtractor`` to ensure instruction and cues can be separated."""
    from pandas.testing import assert_frame_equal

    filename = Path(tmp_dir.name) / "block.txt"
    _create_presentation_logfile(
        tmp_dir.name, BLOCK_PRESENTATION_DATA_WITH_CUE, "block"
    )

    extractor = PresentationBlockExtractor(
        log_or_df=filename,
        block_cue_names=["A", "B"],
        convert_to_seconds=["Time", "Duration"],
        scanner_event_type="Pulse",
        scanner_trigger_code="99",
        rest_block_codes="rest",
        rest_code_frequency="variable",
        split_cue_as_instruction=True,
        quit_code="quit",
        block_cues_without_instruction=("B",),
        drop_instruction_cues=drop_instruction_cues,
    )

    expected_df = pd.DataFrame(
        {
            "onset": [0.0, 1.0, 10.0],
            "duration": [1.0, 9.0, 10.0],
            "trial_type": ["A_instruction", "A", "B"],
            "mean_rt": [float("NaN"), float("NaN"), 0.5],
            "mean_acc": [float("NaN"), 0.0, 1.0],
        }
    )

    onsets = extractor.extract_onsets()
    durations = extractor.extract_durations()
    trial_types = extractor.extract_trial_types()
    response_map = {"hit": 1, "miss": 0}
    mean_rts = extractor.extract_mean_reaction_times(
        response_map=response_map,
        response_type="correct",
    )
    mean_accs = extractor.extract_mean_accuracies(response_map=response_map)

    df = pd.DataFrame(
        {
            "onset": onsets,
            "duration": durations,
            "trial_type": trial_types,
            "mean_rt": mean_rts,
            "mean_acc": mean_accs,
        }
    )

    if drop_instruction_cues:
        expected_df = expected_df.drop([0], axis=0).reset_index(drop=True)

    assert_frame_equal(df, expected_df)


@pytest.mark.parametrize("scanner_start_time", [None, 94621])
def test_PresentationEventExtractor(tmp_dir, scanner_start_time):
    """Test for ``PresentationEventExtractor``."""
    from pandas.testing import assert_frame_equal

    filename = Path(tmp_dir.name) / "event.txt"
    _create_presentation_logfile(tmp_dir.name, EVENT_PRESENTATION_DATA, "event")

    expected_df = pd.DataFrame(
        {
            "onset": [7.9107, 10.8965],
            "duration": [0.5, 0.5],
            "trial_type": ["incongruentright", "congruentleft"],
            "reaction_time": [0.7058, float("nan")],
            "response": ["hit", "miss"],
            "accuracy": [1, 0],
        }
    )

    extractor = PresentationEventExtractor(
        log_or_df=filename,
        trial_types=["incongruent.*", "congruentleft"],
        convert_to_seconds=["Time", "Duration"],
        scanner_event_type="Pulse",
        scanner_trigger_code="99",
    )

    scanner_start_time = scanner_start_time / 10000 if scanner_start_time else None
    onsets = extractor.extract_onsets(scanner_start_time=scanner_start_time)
    durations = extractor.extract_durations()
    trial_types = extractor.extract_trial_types()
    reaction_times = extractor.extract_reaction_times()
    responses = extractor.extract_responses()
    accuracies = extractor.extract_accuracies({"hit": 1, "miss": 0})

    df = pd.DataFrame(
        {
            "onset": onsets,
            "duration": durations,
            "trial_type": trial_types,
            "reaction_time": reaction_times,
            "response": responses,
            "accuracy": accuracies,
        }
    )
    assert_frame_equal(df, expected_df)


def _create_eprime_logfile(dst_dir, data, design):
    dst_dir = Path(dst_dir)

    filename = dst_dir / f"{design}.txt"
    with open(filename, mode="w") as file:
        for line in data:
            file.write("\t".join(line) + "\n")


@pytest.mark.parametrize("offset_column_name", ("Data.OffsetTime", None))
def test_EPrimeBlockExtractor(tmp_dir, offset_column_name):
    """Test for ``EPrimeBlockExtractor``."""
    from pandas.testing import assert_frame_equal

    filename = Path(tmp_dir.name) / "block.txt"

    _create_eprime_logfile(tmp_dir.name, EPRIME_DATA_NO_CUE, "block")

    for column, scanner_start_time in [(None, 10.0), ("TriggerStart", None)]:
        expected_df = pd.DataFrame(
            {
                "onset": [-1.0, 19.0] if column else [0.0, 20.0],
                "duration": [11.0, 11.0] if offset_column_name else [20.0, 20.0],
                "trial_type": ["A", "B"],
            }
        )
        extractor = EPrimeBlockExtractor(
            log_or_df=filename,
            block_cue_names=["A", "B"],
            onset_column_name="Data.OnsetTime",
            procedure_column_name="Procedure",
            trigger_column_name=column,
            convert_to_seconds=["Data.OnsetTime", "TriggerStart", "Data.OffsetTime"],
            n_discarded_volumes=1 if column else 0,
            tr=1 if column else None,
            rest_block_codes="Rest",
            rest_code_frequency="variable",
        )

        onsets = extractor.extract_onsets(scanner_start_time=scanner_start_time)
        durations = extractor.extract_durations(offset_column_name=offset_column_name)
        trial_types = extractor.extract_trial_types()
        df = pd.DataFrame(
            {"onset": onsets, "duration": durations, "trial_type": trial_types}
        )
        assert_frame_equal(df, expected_df)


def test_EPrimeBlockExtractor_mean_rt_and_accuracy(tmp_dir):
    """
    Test for ``extract_mean_reaction_times`` and ``extract_mean_accuracies`` in
    ``EPrimeBlockExtractor``."""
    filename = Path(tmp_dir.name) / "block.txt"
    _create_eprime_logfile(tmp_dir.name, EPRIME_DATA_NO_CUE, "block")

    extractor = EPrimeBlockExtractor(
        log_or_df=filename,
        block_cue_names=["A.*", "B"],
        onset_column_name="Data.OnsetTime",
        procedure_column_name="Procedure",
        trigger_column_name="TriggerStart",
        convert_to_seconds=["Data.OnsetTime", "Data.RT", "TriggerStart"],
        rest_block_codes="Res.*",
        rest_code_frequency="variable",
    )

    # Test mean RT for all trials (default)
    mean_rts_all = extractor.extract_mean_reaction_times(
        reaction_time_column_name="Data.RT",
    )
    assert len(mean_rts_all) == 2
    # Block A: (0.5 + 0.6) / 2 = 0.55
    assert mean_rts_all[0] == pytest.approx(0.55)
    # Block B: (0.5 + 0.7) / 2 = 0.6
    assert mean_rts_all[1] == pytest.approx(0.6)

    # Test mean RT for correct trials only
    mean_rts_correct = extractor.extract_mean_reaction_times(
        reaction_time_column_name="Data.RT",
        subject_response_column="Data.RESP",
        correct_response_column="Data.CRESP",
        response_type="correct",
        response_trial_names=("A", "B"),
    )
    assert len(mean_rts_correct) == 2
    # Block A: only correct trial has RT 0.5
    assert mean_rts_correct[0] == 0.5
    # Block B: both correct, (0.5 + 0.7) / 2 = 0.6
    assert mean_rts_correct[1] == 0.6

    # Test that columns are required when response_type is not "all"
    with pytest.raises(ValueError, match="subject_response_column"):
        extractor.extract_mean_reaction_times(
            reaction_time_column_name="Data.RT",
            response_type="correct",
        )

    mean_accs = extractor.extract_mean_accuracies(
        subject_response_column="Data.RESP",
        correct_response_column="Data.CRESP",
        response_trial_names=("A", "B"),
    )
    assert len(mean_accs) == 2
    # Block A
    assert mean_accs[0] == 0.5
    # Block B
    assert mean_accs[1] == 1.0


@pytest.mark.parametrize(
    "drop_instruction_cues, offset_column_name",
    ([True, "Data.OffsetTime"], [False, None]),
)
def test_EPrimeBlockExtractor_instruction_cue_separation(
    tmp_dir, drop_instruction_cues, offset_column_name
):
    """Test for ``EPrimeBlockExtractor`` to ensure instruction and cues can be separated."""
    from pandas.testing import assert_frame_equal

    filename = Path(tmp_dir.name) / "block.txt"
    _create_eprime_logfile(tmp_dir.name, EPRIME_DATA_WITH_CUE, "block")

    extractor = EPrimeBlockExtractor(
        log_or_df=filename,
        block_cue_names=["A", "B"],
        onset_column_name="Data.OnsetTime",
        procedure_column_name="Procedure",
        trigger_column_name="TriggerStart",
        convert_to_seconds=[
            "Data.OnsetTime",
            "Data.RT",
            "TriggerStart",
            "Data.OffsetTime",
        ],
        rest_block_codes="Rest",
        rest_code_frequency="variable",
        split_cue_as_instruction=True,
        block_cues_without_instruction=("B"),
    )

    expected_df = pd.DataFrame(
        {
            "onset": [0.0, 1.0, 10.0],
            "duration": [1.0, 9.0, 10.0],
            "trial_type": ["A_instruction", "A", "B"],
            "mean_rt": [float("NaN"), float("NaN"), 0.5],
            "mean_acc": [float("NaN"), 0.0, 1.0],
        }
    )

    onsets = extractor.extract_onsets()
    durations = extractor.extract_durations(offset_column_name=offset_column_name)
    trial_types = extractor.extract_trial_types()
    mean_rts = extractor.extract_mean_reaction_times(
        reaction_time_column_name="Data.RT",
        subject_response_column="Data.RESP",
        correct_response_column="Data.CRESP",
        response_type="correct",
    )
    mean_accs = extractor.extract_mean_accuracies(
        subject_response_column="Data.RESP",
        correct_response_column="Data.CRESP",
    )

    df = pd.DataFrame(
        {
            "onset": onsets,
            "duration": durations,
            "trial_type": trial_types,
            "mean_rt": mean_rts,
            "mean_acc": mean_accs,
        }
    )

    if drop_instruction_cues:
        expected_df.drop([0], axis=0).reset_index(inplace=True, drop=True)

    assert_frame_equal(df, expected_df, rtol=1e3)


def test_EPrimeEventExtractor(tmp_dir):
    """Test for ``EPrimeEventExtractor``."""
    from pandas.testing import assert_frame_equal

    filename = Path(tmp_dir.name) / "event.txt"

    _create_eprime_logfile(tmp_dir.name, EPRIME_DATA_NO_CUE, "event")

    expected_df = pd.DataFrame(
        {
            "onset": [0.0, 10.0, 20.0, 30.0, 40.0],
            "duration": [1.0, 1.0, 1.0, 1.0, 1.0],
            "trial_type": ["A", "A", "B", "B", "Rest"],
            "reaction_time": [0.5, 0.6, 0.5, 0.7, float("nan")],
            "response": [1, 0, 1, 1, 1],
        }
    )

    for column, scanner_start_time in [(None, 10.0), ("TriggerStart", None)]:
        extractor = EPrimeEventExtractor(
            log_or_df=filename,
            trial_types=["A", "B", "Rest"],
            onset_column_name="Data.OnsetTime",
            procedure_column_name="Procedure",
            trigger_column_name=column,
            convert_to_seconds=[
                "Data.OnsetTime",
                "Data.OffsetTime",
                "Data.RT",
                "TriggerStart",
            ],
        )

        onsets = extractor.extract_onsets(scanner_start_time=scanner_start_time)
        durations = extractor.extract_durations(offset_column_name="Data.OffsetTime")
        trial_types = extractor.extract_trial_types()
        reaction_times = extractor.extract_reaction_times(
            reaction_time_column_name="Data.RT"
        )
        responses = extractor.extract_accuracies(
            subject_response_column="Data.RESP",
            correct_response_column="Data.CRESP",
        )

        df = pd.DataFrame(
            {
                "onset": onsets,
                "duration": durations,
                "trial_type": trial_types,
                "reaction_time": reaction_times,
                "response": responses,
            }
        )
        assert_frame_equal(df, expected_df)


def test_add_instruction_timing():
    """Test for ``add_instruction_timing``."""
    from pandas.testing import assert_frame_equal

    events_df = pd.DataFrame(
        {
            "onset": [0.0, 10.0, 20.0],
            "duration": [10.0, 10.0, 10.0],
            "trial_type": ["Face", "Place", "Rest"],
            "random": [1, 1, 1],
        }
    )

    new_events_df = add_instruction_timing(
        events_df,
        instruction_duration=2.0,
        exclude_trial_types=["Rest"],
    )

    expected_df = pd.DataFrame(
        {
            "onset": [0.0, 2.0, 10.0, 12.0, 20.0],
            "duration": [2.0, 8.0, 2.0, 8.0, 10.0],
            "trial_type": [
                "Face_instruction",
                "Face",
                "Place_instruction",
                "Place",
                "Rest",
            ],
            "random": [float("nan"), 1, float("nan"), 1, 1],
        }
    )

    assert_frame_equal(new_events_df, expected_df)

    events_df_missing = pd.DataFrame(
        {
            "onset": [0.0, 10.0],
            "trial_type": ["Face", "Place"],
        }
    )

    with pytest.raises(ValueError, match="must contain"):
        add_instruction_timing(events_df_missing, instruction_duration=2.0)
