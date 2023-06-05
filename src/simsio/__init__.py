from configparser import ConfigParser

__version__ = "0.2.2"

RC_FILE = ".simsiorc"
rc = ConfigParser(interpolation=None)
rc.read(RC_FILE)

from simsio.simulations import *
from simsio.runsim import *
