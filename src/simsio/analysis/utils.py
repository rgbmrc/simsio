import unicodedata
import hashlib  # noqa: F401
import re

import numpy as np

__all__ = ["is_numeric", "as_ndarray"]


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def sanitize_path(s):
    # TODO platform dependent, improve, see https://stackoverflow.com/q/295135
    # FIXME name-collisions possible due to 128 char limit

    return s.replace(":", "!").replace("/", "_")[:128]


def is_numeric(val):
    return np.issubdtype(np.asanyarray(val).dtype, np.number)


def as_ndarray(x):
    try:
        return x.to_masked_array()  # xarray.DataArray
    except AttributeError:
        # does not respect xarray's hideous nan mask
        return np.asanyarray(x)  # anything else
