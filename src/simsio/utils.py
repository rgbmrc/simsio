import logging
from importlib import import_module

from simsio.settings import rc

__all__ = ["get_module_attr"]


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


def get_module_attr(name):
    mod, attr = name.rsplit(".", 1)
    return getattr(import_module(mod), attr)


def as_scalar(x):
    if hasattr(x, "item"):
        return x.item()
    return x
