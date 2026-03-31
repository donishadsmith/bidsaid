"""Decorator functions."""

import functools, inspect

from typing import Any, Callable

from ._helpers import iterable_to_str
from .io import get_nifti_header


def check_all_none(parameter_names: list[str]) -> Callable:
    """
    Checks if specific parameters are assigned ``None``.

    Parameters
    ----------
    parameter_names : :obj:`list[str]`
        List of parameter names to check.

    Returns
    -------
    Callable
        Decorator function wrapping target function.
    """

    def decorator(func: Callable) -> Callable:
        signature = inspect.signature(func)
        if invalid_params := [
            param
            for param in parameter_names
            if param not in signature.parameters.keys()
        ]:
            raise NameError(
                "Error in ``parameter_names`` of decorator. The following "
                f"parameters are not in the signature of '{func.__name__}': "
                f"{iterable_to_str(invalid_params)}."
            )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            all_param_values = [bound_args.arguments[name] for name in parameter_names]
            if all(value is None for value in all_param_values):
                raise ValueError(
                    "All of the following arguments cannot be None, "
                    f"one must be specified: {iterable_to_str(parameter_names)}."
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def check_nifti(nifti_param_name: str | None = None) -> Callable:
    """
    Checks if input NIfTI has qform or sform codes set to scanner.

    Parameters
    ----------
    nifti_param_name : :obj:`list[str]`
        Name of the NIfTI parameter. If None, assumes the
        NIfTI parameter is "nifti_file_or_img".

    Returns
    -------
    Callable
        Decorator function wrapping target function.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            param_name = (
                "nifti_file_or_img" if not nifti_param_name else nifti_param_name
            )
            img_val = bound_args.arguments.get(param_name)
            if img_val:
                hdr = get_nifti_header(img_val)

                # sform takes precedence over qform
                code = "sform_code" if hdr["sform_code"] else "qform_code"
                if hdr[code] != 1:
                    raise ValueError(
                        f"The {code} is not set to 'scanner' and is not a raw NIfTI image."
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator
