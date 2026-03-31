"""Custom exceptions."""

from typing import Literal


class SliceAxisError(Exception):
    """
    Incorrect slice axis.

    Raised when the number of slices does not match "slice_end" plus one.

    Parameters
    ----------
    slice_axis : :obj:`Literal["x", "y", "z"]`
        The specified slice dimension.

    n_slices : :obj:`int`
        The number of slices from the specified ``slice_axis``.

    slice_end : :obj:`int`
        The number of slices specified by "slice_end" in the NIfTI header.

    message : :obj:`str` or :obj:`None`:
        The error message. If None, a default error message is used.
    """

    def __init__(
        self,
        slice_axis: Literal["x", "y", "z"],
        n_slices: int,
        slice_end: int,
        message: str | None = None,
    ):
        if not message:
            self.message = (
                "Incorrect slice axis. Number of slices for "
                f"{slice_axis} dimension is {n_slices} but "
                f"'slice_end' in NIfTI header is {slice_end}."
            )
        else:
            self.message = message

        super().__init__(self.message)


class DataDimensionError(Exception):
    """
    Incorrect data dimensionality.
    """

    pass


class PathDoesNotExist(Exception):
    """Exception when path does not exist."""

    def __init__(self, path):
        self.message = f"The following path does not exist: {path}"

        super().__init__(self.message)
