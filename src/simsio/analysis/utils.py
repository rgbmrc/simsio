import numpy as np

__all__ = ["is_numeric", "as_ndarray"]


def is_numeric(val):
    return np.issubdtype(np.asanyarray(val).dtype, np.number)


def as_ndarray(x):
    try:
        return x.to_masked_array()  # xarray.DataArray
    except AttributeError:
        # does not respect xarray's hideous nan mask
        return np.asanyarray(x)  # anything else
