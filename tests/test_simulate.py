from pathlib import Path

import nibabel as nib, pytest

from nifti2bids.simulate import (
    simulate_nifti_image,
    simulate_bids_dataset,
)


def test_simulate_nifti_image():
    """Test for ``simulate_nifti_image``."""
    img = simulate_nifti_image(img_shape=(20, 20, 20, 20))
    assert isinstance(img, nib.Nifti1Image)

    with pytest.raises(ValueError):
        simulate_nifti_image((10, 10))


@pytest.mark.parametrize("n_sessions", [1, None])
def test_simulate_bids_dataset(tmp_dir, n_sessions):
    """Test for ``simulate_bids_dataset``."""
    import bids

    bids_root = simulate_bids_dataset(
        output_dir=Path(tmp_dir.name) / "BIDS", n_sessions=n_sessions
    )

    layout = bids.BIDSLayout(bids_root, derivatives=True)
    files = layout.get(return_type="file", extension="nii.gz")
    assert len(files) == 2

    if n_sessions:
        assert layout.get_sessions() == ["1"]
    else:
        assert not layout.get_sessions()
