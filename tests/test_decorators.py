import nibabel as nib
import pytest

from bidsaid._decorators import check_all_none, check_nifti
from bidsaid.simulate import simulate_nifti_img


def test_check_all_none():
    """Test ``check_all_none`` decorator"""

    def mock_func(a=None, b=None):
        return a

    with pytest.raises(NameError):
        check_all_none(parameter_names=["a", "c"])(mock_func)

    with pytest.raises(ValueError):
        check_all_none(parameter_names=["a", "b"])(mock_func)(None, None)

    # Should capture positional and keyword args
    assert check_all_none(parameter_names=["a", "b"])(mock_func)(True, b=None)


def test_check_nifti():
    """Test ``check_nifti`` decorator"""
    img = simulate_nifti_img((10, 10, 10, 10))
    img.header["qform_code"] = 0
    img.header["sform_code"] = 1

    @check_nifti("nifti_img")
    def mock_func_a(nifti_img):
        return nifti_img

    assert isinstance(mock_func_a(img), nib.nifti1.Nifti1Image)

    @check_nifti()
    def mock_func_b(nifti_file_or_img):
        return nifti_file_or_img

    assert isinstance(mock_func_b(img), nib.nifti1.Nifti1Image)

    with pytest.raises(ValueError):
        img.header["qform_code"] = 0
        img.header["sform_code"] = 0
        mock_func_b(img)
