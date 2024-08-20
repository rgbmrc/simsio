from configparser import ConfigParser

__version__ = "0.3.0"

RC_FILE = ".simsiorc"
rc = ConfigParser(interpolation=None)
# use read_file instead?
# https://docs.python.org/3/library/configparser.html#configparser.ConfigParser.read
rc.read(RC_FILE)

from simsio.simulations import *
from simsio.runsim import *
