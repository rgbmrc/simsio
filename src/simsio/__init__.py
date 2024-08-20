from configparser import ConfigParser
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

__version__ = "0.3.0"

RC_FILE = ".simsiorc"
rc = ConfigParser(interpolation=None)
# use read_file instead?
# https://docs.python.org/3/library/configparser.html#configparser.ConfigParser.read
rc.read(RC_FILE)


@contextmanager
def rc_context(filenames):
    with NamedTemporaryFile("w+") as f:
        rc.write(f)
        f.flush()
        rc.read(filenames)
        yield
        rc.clear()
        assert f.name in rc.read(f.name)


from simsio.simulations import *
from simsio.runsim import *
