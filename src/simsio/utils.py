import logging
import logging.config
from importlib import import_module
from math import isclose

from simsio.settings import rc

__all__ = ["get_module_attr", "as_scalar", "as_int", "setup_logging"]


def get_module_attr(name):
    mod, attr = name.rsplit(".", 1)
    return getattr(import_module(mod), attr)


def as_scalar(x):
    if hasattr(x, "item"):
        return x.item()
    return x


def as_int(x) -> int:
    int_x = round(as_scalar(x))
    assert isclose(int_x, x), f"{x} is not integer-like"
    return int_x


def setup_logging(path):
    """Setup logging."""

    handlers = {
        "term": {
            "class": logging.StreamHandler,
            "stream": "ext://sys.stdout",
            "formatter": "fmt",
        },
        "file": {
            "class": logging.FileHandler,
            "filename": path,
            "formatter": "fmt",
        },
    }
    loggingrc = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"fmt": rc["logging-format"]},
        "handlers": handlers,
        "loggers": {k: {"level": l} for k, l in rc["logging-levels"].items()},
        "root": {"handlers": handlers},
    }
    logging.config.dictConfig(loggingrc)
    logging.captureWarnings(True)
