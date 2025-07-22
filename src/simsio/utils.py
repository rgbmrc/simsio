from importlib import import_module

import numpy as np


def get_module_attr(name):
    mod, attr = name.rsplit(".", 1)
    return getattr(import_module(mod), attr)


def is_numeric(val):
    return np.issubdtype(np.asanyarray(val).dtype, np.number)
