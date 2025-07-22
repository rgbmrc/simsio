import numpy as np

__all__ = ["is_numeric"]


def is_numeric(val):
    return np.issubdtype(np.asanyarray(val).dtype, np.number)
