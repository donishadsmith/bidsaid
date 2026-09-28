import io
from pathlib import Path

import numpy as np, pandas as pd, pytest

from bidsaid.qc import (
    compute_n_dummy_scans,
    compute_framewise_displacement,
    create_censor_mask,
    merge_censor_masks,
    compute_consecutive_censor_stats,
    compute_global_signal,
    create_spike_regressors,
    get_n_censored_volumes,
    downsample_img,
    upsample_img,
)


def _create_data(tmp_dir, data, input_type, has_header=True):
    data = io.StringIO("".join(data), newline=None)

    df = pd.read_csv(data, sep="\t")
    if input_type == "file":
        input_data = Path(tmp_dir.name) / "confounds.tsv"
        df.to_csv(input_data, sep="\t", index=None, header=has_header)
    else:
        input_data = df

    return input_data


@pytest.mark.parametrize("input_type", ["file", "df"])
def test_compute_n_dummy_scans(tmp_dir, input_type):
    "Test for ``compute_n_dummy_scans``."
    data = ["non_steady_state_outlier_01\tnon_steady_state_outlier_02\n", "0\t0\n"]
    assert compute_n_dummy_scans(_create_data(tmp_dir, data, input_type)) == 2


@pytest.mark.parametrize("input_type", ["file", "df"])
def test_compute_framewise_displacement(tmp_dir, input_type):
    "Test ``compute_framewise_displacement``."
    header = (
        "\t".join(["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]) + "\n"
    )
    data = [header, "1\t1\t1\t1\t1\t1\n", "1\t1\t1\t1\t1\t1\n"]

    input_data = _create_data(tmp_dir, data, input_type)
    assert np.array_equal(compute_framewise_displacement(input_data), np.array([0, 0]))

    if input_type == "file":
        input_data = _create_data(
            tmp_dir,
            data,
            input_type,
            has_header=False,
        )
        assert np.array_equal(
            compute_framewise_displacement(
                input_data,
                has_header=False,
                verbose=True,
            ),
            np.array([0, 0]),
        )
    else:
        input_data = _create_data(tmp_dir, data, input_type)
        assert np.array_equal(
            compute_framewise_displacement(
                input_data.to_numpy(copy=True),
                verbose=True,
            ),
            np.array([0, 0]),
        )


@pytest.mark.parametrize("input_type", ["file", "df"])
def test_create_censor_mask(tmp_dir, input_type):
    data = ["fd\tplaceholder\n", "0\n0.1\n0.5\n0.6\n0.7\n0.1\n"]

    censor_mask = create_censor_mask(
        _create_data(tmp_dir, data, input_type), n_dummy_scans=2
    )
    assert np.array_equal(censor_mask, np.array([0, 0, 1, 1, 1, 1]))

    censor_mask = create_censor_mask(
        _create_data(tmp_dir, data, input_type), n_dummy_scans=2
    )
    assert np.array_equal(censor_mask, np.array([0, 0, 1, 1, 1, 1]))

    if input_type == "file":
        censor_mask = create_censor_mask(
            _create_data(tmp_dir, data, input_type),
            n_dummy_scans=2,
            column_name="fd",
            threshold=0.5,
            verbose=True,
        )
        assert np.array_equal(censor_mask, np.array([0, 0, 1, 0, 0, 1]))
    else:
        data = _create_data(tmp_dir, data, input_type)
        censor_mask = create_censor_mask(
            data["fd"].to_numpy(copy=True),
            n_dummy_scans=2,
            threshold=0.5,
            verbose=True,
        )
        assert np.array_equal(censor_mask, np.array([0, 0, 1, 0, 0, 1]))


def test_merge_censor_mask():
    "Test ``merge_censor_mask``."
    censor_masks = [np.array([0, 0, 1]), np.array([1, 0, 1])]

    assert np.array_equal(merge_censor_masks(censor_masks), np.array([0, 0, 1]))


def test_compute_consecutive_censor_stats():
    "Test ``compute_consecutive_censor_stats``."
    censor_mask = np.array([0, 0, 1, 1])
    mean, std = compute_consecutive_censor_stats(censor_mask).values()
    assert (mean, std) == (2.0, 0.0)

    mean, std = compute_consecutive_censor_stats(censor_mask, n_dummy_scans=2).values()
    assert (mean, std) == (0.0, np.nan)


def test_compute_global_signal():
    "Test for ``compute_global_signal``."
    from bidsaid.simulate import simulate_nifti_image

    nifti_img = simulate_nifti_image((3, 3, 3, 10))

    result = compute_global_signal(nifti_img)

    assert set(result.keys()) == {
        "global_signal",
        "global_signal_percent_change",
    }

    assert result["global_signal"].shape == (10,)
    assert result["global_signal_percent_change"].shape == (10,)


def test_create_spike_regressors():
    """Test for ``create_spike_regressors``"""
    from pandas.testing import assert_frame_equal

    arr = np.array([0, 1, 1, 0])
    df = create_spike_regressors(arr)
    expected_df = pd.DataFrame(
        np.array([[1, 0, 0, 0], [0, 0, 0, 1]]).T,
        columns=["spike_regressor_0", "spike_regressor_1"],
    )
    assert_frame_equal(expected_df, df)

    arr = np.array([1, 1, 1, 1])
    df = create_spike_regressors(arr)
    assert df.empty


def test_get_n_censored_volumes(tmp_dir):
    """Test for ``get_n_censored_volumes``"""
    arr = np.array([1, 0, 0, 1])
    assert get_n_censored_volumes(arr) == 2

    file = Path(tmp_dir.name) / "cens.1D"
    np.savetxt(file, arr)
    assert get_n_censored_volumes(file) == 2


def test_interpolation(nifti_img_and_path):
    import numpy as np

    """Test for ``upsample_img`` and ``downsample_img``"""
    img, _ = nifti_img_and_path

    upsampled_image = upsample_img(img, 2)
    # should be 4 gaps, 2 new frames = 8 + 5 original frames = 13
    # tr = 1 / (2 + 1)
    assert upsampled_image.shape[-1] == 13
    np.testing.assert_allclose(
        upsampled_image.header.get_zooms()[-1], 0.33333, rtol=1e-4
    )

    downsampled_image = downsample_img(upsampled_image, 2)
    assert downsampled_image.shape[-1] == 5
    np.testing.assert_allclose(
        downsampled_image.header.get_zooms()[-1], 0.99
    )  # close enough

    np.testing.assert_allclose(img.get_fdata(), downsampled_image.get_fdata())
