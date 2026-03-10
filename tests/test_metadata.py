import nibabel as nib, numpy as np, pytest
import nifti2bids.metadata as bids_meta


@pytest.mark.parametrize("return_header", (False, True))
def test_get_hdr_metadata(nifti_img_and_path, return_header):
    """Test for ``get_hdr_metadata``."""
    img, _ = nifti_img_and_path
    img.header["slice_end"] = 100

    if return_header:
        slice_end, hdr = bids_meta.get_hdr_metadata(
            metadata_name="slice_end",
            nifti_file_or_img=img,
            return_header=return_header,
        )
        assert isinstance(hdr, nib.nifti1.Nifti1Header)
    else:
        slice_end = bids_meta.get_hdr_metadata(
            metadata_name="slice_end",
            nifti_file_or_img=img,
            return_header=return_header,
        )

    assert slice_end == 100


def test_determine_slice_axis(nifti_img_and_path):
    """Test for ``determine_slice_axis``."""
    img, _ = nifti_img_and_path

    with pytest.raises(ValueError):
        bids_meta.determine_slice_axis(img)

    # Subtract one to convert to index
    img.header["slice_end"] = img.get_fdata().shape[2] - 1
    assert bids_meta.determine_slice_axis(img) == 2


def test_get_n_volumes(nifti_img_and_path):
    """Test for ``get_n_volumes``."""
    img, _ = nifti_img_and_path
    assert bids_meta.get_n_volumes(img) == 5

    from nifti2bids.simulate import simulate_nifti_image
    from nifti2bids._exceptions import DataDimensionError

    with pytest.raises(DataDimensionError):
        bids_meta.get_n_volumes(simulate_nifti_image((10, 10, 10)))


def test_get_image_orientation(nifti_img_and_path):
    """Test for ``get_image_orientation``"""
    img, _ = nifti_img_and_path
    orientation_dict, orientation = bids_meta.get_image_orientation(img)
    assert orientation_dict == {"i": "L -> R", "j": "P -> A", "k": "I -> S"}
    assert orientation == ("R", "A", "S")


def test_get_n_slices(nifti_img_and_path):
    """Test for ``get_n_slices``."""
    from nifti2bids._exceptions import SliceAxisError

    img, _ = nifti_img_and_path
    # Subtract one to convert to index
    img.header["slice_end"] = img.get_fdata().shape[2] - 1

    with pytest.raises(SliceAxisError):
        bids_meta.get_n_slices(img, slice_axis="i")

    assert bids_meta.get_n_slices(img, slice_axis="k") == img.header["slice_end"] + 1
    assert bids_meta.get_n_slices(img) == img.header["slice_end"] + 1


def test_get_tr(nifti_img_and_path):
    """Test for ``get_tr``."""
    img, _ = nifti_img_and_path
    img.header["pixdim"][4] = 2.3
    assert bids_meta.get_tr(img) == 2.3
    assert not isinstance(bids_meta.get_tr(img), np.floating)
    assert not isinstance(bids_meta.get_tr(img), np.integer)

    img.header["pixdim"][4] = 0
    with pytest.raises(ValueError):
        bids_meta.get_tr(img)


@pytest.mark.parametrize("slice_acquisition_method", ("sequential", "interleaved"))
def test_create_slice_timing_singleband(slice_acquisition_method):
    """Test for ``create_slice_timing`` for singleband acquisition."""
    from nifti2bids.simulate import simulate_nifti_image

    img = simulate_nifti_image((10, 10, 4, 10))
    img.header["pixdim"][4] = 2
    img.header["slice_end"] = 3

    if slice_acquisition_method == "sequential":
        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=True,
        )
        assert slice_timing_list == [0, 0.5, 1, 1.5]

        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=False,
        )
        assert slice_timing_list == [1.5, 1, 0.5, 0]
    else:
        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=True,
            interleaved_pattern="odd",
        )
        assert slice_timing_list == [0, 1, 0.5, 1.5]

        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=False,
            interleaved_pattern="odd",
        )
        assert slice_timing_list == [1.5, 0.5, 1, 0]

        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=True,
            interleaved_pattern="even",
        )
        assert slice_timing_list == [1, 0, 1.5, 0.5]

        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=False,
            interleaved_pattern="even",
        )
        assert slice_timing_list == [0.5, 1.5, 0, 1]

        with pytest.raises(ValueError):
            slice_timing_list = bids_meta.create_slice_timing(
                nifti_file_or_img=img,
                slice_acquisition_method=slice_acquisition_method,
                ascending=False,
                interleaved_pattern="incorrect_value",
            )


def test_philips_interleaved():
    from nifti2bids.simulate import simulate_nifti_image

    img = simulate_nifti_image((10, 10, 9, 10))
    img.header["pixdim"][4] = 2
    img.header["slice_end"] = 8

    slice_timing_list = bids_meta.create_slice_timing(
        nifti_file_or_img=img,
        slice_acquisition_method="interleaved",
        ascending=True,
        interleaved_pattern="philips",
    )
    # slice order: 0, 3, 6, 1, 4, 7, 2, 5, 8
    assert np.allclose(
        np.array(slice_timing_list),
        np.array(
            [
                0,
                0.66,
                1.33,
                0.22,
                0.88,
                1.55,
                0.44,
                1.11,
                1.77,
            ]
        ),
        atol=1e-2,
    )

    slice_timing_list = bids_meta.create_slice_timing(
        nifti_file_or_img=img,
        slice_acquisition_method="interleaved",
        ascending=False,
        interleaved_pattern="philips",
    )
    # slice order: 8, 5, 2, 7, 4, 1, 6, 3, 0
    assert np.allclose(
        np.array(slice_timing_list),
        np.array(
            [
                1.77,
                1.11,
                0.44,
                1.55,
                0.88,
                0.22,
                1.33,
                0.66,
                0,
            ]
        ),
        atol=1e-2,
    )


@pytest.mark.parametrize("slice_acquisition_method", ("central", "reversed_central"))
def test_central_and_reversed_central(slice_acquisition_method):
    """
    Test for ``create_slice_timing`` for singleband acquisition
    with the "central" and "reversed_central" order.
    """
    from nifti2bids.simulate import simulate_nifti_image

    img = simulate_nifti_image((10, 10, 5, 10))
    img.header["pixdim"][4] = 2
    img.header["slice_end"] = 4

    slice_timing_list = bids_meta.create_slice_timing(
        nifti_file_or_img=img,
        slice_acquisition_method=slice_acquisition_method,
    )

    slice_timing_list = [round(x, 2) for x in slice_timing_list]
    if slice_acquisition_method == "central":
        assert slice_timing_list == [1.2, 0.4, 0, 0.8, 1.6]
    else:
        assert slice_timing_list == [0, 0.8, 1.6, 1.2, 0.4]

    img = simulate_nifti_image((10, 10, 6, 10))
    img.header["pixdim"][4] = 2
    img.header["slice_end"] = 5

    slice_timing_list = bids_meta.create_slice_timing(
        nifti_file_or_img=img,
        slice_acquisition_method=slice_acquisition_method,
    )

    slice_timing_list = [round(x, 2) for x in slice_timing_list]
    if slice_acquisition_method == "central":
        assert slice_timing_list == [1.67, 1.00, 0.33, 0, 0.67, 1.33]
    else:
        assert slice_timing_list == [0, 0.67, 1.33, 1.67, 1.00, 0.33]


@pytest.mark.parametrize("slice_acquisition_method", ("sequential", "interleaved"))
def test_create_slice_timing_multiband(slice_acquisition_method):
    """Test for ``create_slice_timing`` for multiband acquisition."""
    from nifti2bids.simulate import simulate_nifti_image

    img = simulate_nifti_image((12, 12, 10, 12))
    img.header["pixdim"][4] = 2
    img.header["slice_end"] = 9

    if slice_acquisition_method == "sequential":
        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=True,
            multiband_factor=2,
        )
        assert np.allclose(
            slice_timing_list, [0.0, 0.4, 0.8, 1.2, 1.6, 0.0, 0.4, 0.8, 1.2, 1.6]
        )

        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=False,
            multiband_factor=2,
        )
        assert np.allclose(
            slice_timing_list, [1.6, 1.2, 0.8, 0.4, 0.0, 1.6, 1.2, 0.8, 0.4, 0.0]
        )
    else:
        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=True,
            interleaved_pattern="odd",
            multiband_factor=2,
        )
        assert np.allclose(
            slice_timing_list, [0.0, 1.2, 0.4, 1.6, 0.8, 0.0, 1.2, 0.4, 1.6, 0.8]
        )

        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            interleaved_pattern="odd",
            ascending=False,
            multiband_factor=2,
        )
        assert np.allclose(
            slice_timing_list, [0.8, 1.6, 0.4, 1.2, 0.0, 0.8, 1.6, 0.4, 1.2, 0.0]
        )

        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=True,
            interleaved_pattern="even",
            multiband_factor=2,
        )
        assert np.allclose(
            slice_timing_list, [0.8, 0.0, 1.2, 0.4, 1.6, 0.8, 0.0, 1.2, 0.4, 1.6]
        )

        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            interleaved_pattern="even",
            ascending=False,
            multiband_factor=2,
        )
        assert np.allclose(
            slice_timing_list, [1.6, 0.4, 1.2, 0.0, 0.8, 1.6, 0.4, 1.2, 0.0, 0.8]
        )

        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=True,
            interleaved_pattern="philips",
            multiband_factor=2,
        )
        assert np.allclose(
            slice_timing_list, [0.0, 0.8, 1.6, 0.4, 1.2, 0.0, 0.8, 1.6, 0.4, 1.2]
        )

        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            ascending=False,
            interleaved_pattern="philips",
            multiband_factor=2,
        )
        assert np.allclose(
            slice_timing_list, [0.4, 1.6, 0.8, 0.0, 1.2, 0.4, 1.6, 0.8, 0.0, 1.2]
        )

    with pytest.raises(ValueError):
        slice_timing_list = bids_meta.create_slice_timing(
            nifti_file_or_img=img,
            slice_acquisition_method=slice_acquisition_method,
            multiband_factor=3,
        )


def test_is_3d_img(nifti_img_and_path):
    """Test for ``is_3d_img``."""
    from nifti2bids.simulate import simulate_nifti_image

    img = simulate_nifti_image((10, 10, 10))
    assert bids_meta.is_3d_img(img)

    img, _ = nifti_img_and_path
    assert not bids_meta.is_3d_img(img)


def test_get_scanner_info(nifti_img_and_path):
    """Test for ``get_scanner_info``."""
    img, _ = nifti_img_and_path
    with pytest.raises(ValueError):
        bids_meta.get_scanner_info(img)

    img.header["descrip"] = "Philips Ingenia Elition X 5.7.1"
    manufacturer_name, model_name = bids_meta.get_scanner_info(img)
    assert manufacturer_name == "Philips"
    assert model_name == "Ingenia Elition X 5.7.1"


def test_is_valid_date():
    """Test for ``is_valid_date``."""
    assert bids_meta.is_valid_date("241010", "%y%m%d")


def test_parse_date_from_path():
    """Test for ``parse_date_from_path``."""
    date = bids_meta.parse_date_from_path("101_240820_mprage_32chan.nii", "%y%m%d")
    assert date == "240820"

    date = bids_meta.parse_date_from_path("101_mprage_32chan.nii", "%y%m%d")
    assert not date

    date = bids_meta.parse_date_from_path(r"Users/users/Documents/101_240820", "%y%m%d")
    assert date == "240820"


def test_get_file_creation_date(tmp_dir):
    """Test for ``get_file_creation_date``."""
    from pathlib import Path

    path = Path(tmp_dir.name) / "test.txt"
    with open(path, "w") as f:
        pass

    date = bids_meta.get_file_creation_date(path, "%Y-%m-%d")
    assert bids_meta.is_valid_date(date, "%Y-%m-%d")


@pytest.mark.parametrize(
    "task_volumes", ({5: "flanker", 10: "nback"}, {"flanker": 5, "nback": 10})
)
def test_infer_task_from_image(nifti_img_and_path, task_volumes):
    """Test for ``infer_task_from_image``."""
    img, _ = nifti_img_and_path

    assert bids_meta.infer_task_from_image(img, task_volumes) == "flanker"


def test_get_recon_matrix_pe(nifti_img_and_path):
    """Test for ``get_recon_matrix_pe``."""
    img, _ = nifti_img_and_path

    with pytest.raises(ValueError):
        bids_meta.get_recon_matrix_pe(img, "i")

    img.header["sform_code"] = 1

    assert bids_meta.get_recon_matrix_pe(img, "i") == 20

    assert bids_meta.get_recon_matrix_pe(img, "j") == 20

    assert bids_meta.get_recon_matrix_pe(img, "k") == 10


def test_compute_effective_echo_spacing():
    """Quick assertion test for ``compute_effective_echo_spacing``"""
    water_fat_shift_px = 1.745
    epi_factor = 27

    assert bids_meta.compute_effective_echo_spacing(1.745, 27) == water_fat_shift_px / (
        434.215 * (epi_factor + 1)
    )


def test_compute_total_readout_time():
    """Quick assertion test for ``compute_total_readout_time``"""
    assert bids_meta.compute_total_readout_time(0.005, 96) == 0.005 * 95
    assert bids_meta.compute_total_readout_time(use_fallback_trt=True) == 0.03125

    with pytest.raises(ValueError):
        bids_meta.compute_total_readout_time()


def test_needs_resampling():
    """Test for ``needs_resampling``"""
    from nifti2bids.simulate import simulate_nifti_image

    img_1 = simulate_nifti_image((97, 115, 98, 50))
    assert not bids_meta.needs_resampling(img_1, img_1)

    img_2 = simulate_nifti_image((97, 115, 90, 50))
    assert bids_meta.needs_resampling(img_1, img_2)


def test_direction_to_voxel_axis():
    """Test for ``direction_to_voxel_axis``"""
    from nifti2bids.simulate import simulate_nifti_image

    nifti_img = simulate_nifti_image(img_shape=(5, 5, 5, 5))
    axis, pos = bids_meta.direction_to_voxel_axis(nifti_img, ("A", "P"))
    assert axis == "j"
    assert pos == 1
