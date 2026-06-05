import nibabel as nib, pytest
import bidsaid.io as bids_io


def test_load_nifti(nifti_img_and_path):
    """Test for ``load_nifti``."""
    img, img_path = nifti_img_and_path
    assert isinstance(bids_io.load_nifti(img), nib.nifti1.Nifti1Image)
    assert isinstance(bids_io.load_nifti(img_path), nib.nifti1.Nifti1Image)


def test_regex_glob(nifti_img_and_path):
    """Test for ``regex_glob``"""
    _, img_path = nifti_img_and_path
    files = bids_io.regex_glob(img_path.parent, pattern=r"^.*\.nii")
    assert len(list(files)) == 1


def test_get_nifti_header(nifti_img_and_path):
    """Test for ``get_nifti_header``."""
    img, _ = nifti_img_and_path
    assert isinstance(bids_io.get_nifti_header(img), nib.nifti1.Nifti1Header)


def test_get_nifti_affine(nifti_img_and_path):
    """Test for ``get_nifti_affine``."""
    img, _ = nifti_img_and_path
    assert bids_io.get_nifti_affine(img).shape == (4, 4)


def test_replace_ext():
    """Test for ``replace_ext``."""
    from pathlib import Path

    assert bids_io.replace_ext("file.nii.gz", "json") == Path("file.json")


def test_is_nifti_truncated(nifti_img_and_path):
    """Test for ``is_nifti_truncated``."""
    _, img_path = nifti_img_and_path

    assert bids_io.is_nifti_truncated(img_path) is False

    data = img_path.read_bytes()
    truncated_path = img_path.parent / "truncated.nii"
    truncated_path.write_bytes(data[:-1000])
    assert bids_io.is_nifti_truncated(truncated_path) is True

    with pytest.raises(ValueError):
        bids_io.is_nifti_truncated("test.nii.gz")


@pytest.mark.parametrize("compression_level", [1, 6, 9])
def test_compress_image(nifti_img_and_path, compression_level):
    """Test for ``compress_image``."""
    import numpy as np

    img, img_path = nifti_img_and_path

    files = list(img_path.parent.glob("*"))
    assert len(files) == 1
    assert files[0].suffix == ".nii"

    bids_io.compress_image(
        img_path, remove_src_file=True, compression_level=compression_level
    )

    files = list(img_path.parent.glob("*"))
    assert len(files) == 1
    assert files[0].suffixes == [".nii", ".gz"]

    compressed_img = bids_io.load_nifti(files[0])
    assert compressed_img.shape == img.shape
    np.testing.assert_array_equal(compressed_img.get_fdata(), img.get_fdata())
    assert bids_io.get_nifti_header(compressed_img) == bids_io.get_nifti_header(img)
