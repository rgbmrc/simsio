from configparser import ConfigParser
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

__all__ = ["rc", "rc_context"]

RC_FILE = ".simsiorc"
rc = ConfigParser(interpolation=None)
# use read_file instead?
# https://docs.python.org/3/library/configparser.html#configparser.ConfigParser.read
rc.read(RC_FILE)


def check_rc():
    """Raises ValueError if the rc lacks what every simulation needs."""
    if "meta" not in rc["IO-handlers"]:
        raise ValueError("the rc needs a `meta` entry in [IO-handlers]")


@contextmanager
def rc_context(filenames, reset: None | bool = None):
    with NamedTemporaryFile("w+") as f:
        rc.write(f)
        f.flush()
        # in case there are keys that would not get overwritten
        if reset is not False:  # True or None
            rc.clear()
        if reset is None:
            rc.read(RC_FILE)
        rc.read(filenames)
        yield
        rc.clear()
        assert f.name in rc.read(f.name)
