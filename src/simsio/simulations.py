"""**Abbreviations**

 - sim: simulation
 - uid: simulation identifier (human provided key or computer generated uuid)

**Storage elements** associated with a Simulation instance

 name | fullname | I/O | typ | description
 ---------------------------------------------------------------------
 log  | log      |   O | txt | cronological summary of the simulation
 res  | results  |   O | bin | measurements to be tabulated or plotted
 par  | params   | I/O | txt | options used by routines and classes
 dat  | data     | I/O | bin | any persistent object, all the above included
 cfg  | config   | I   | txt | subset of par provided as user input

"""

import logging
import logging.config
import shlex
import sys
import time
import uuid
from cmath import isnan  # cmath just to be extra safe
from functools import wraps, partial
from string import Template
from subprocess import run

import dictdiffer
import dpath
from numpy.ma import masked  # numpy dependency :(

from simsio.configs import cfg_load, cfg_update_uid, SimsQuery
from simsio.iocore import Cache
from simsio.settings import rc
from simsio.utils import as_scalar

__all__ = [
    "Simulation",
    "get_sim",
    "sim_like_arg",
    "sims_iter_like_arg",
    "purge_registry",
    "purge_caches",
    "valid_uuid",
    "sim_registry",
]

logger = logging.getLogger(__name__)

sim_registry = {}


def valid_uuid(uid=None, raise_invalid=False):
    """Returns and/or check the validity of a UUID (universally unique
    identifier).

    Parameters
    ----------
    uid : str, optional
        Input UUID, by default None
    raise_invalid : bool, optional
        Whether ValueError should be raised when uid is not a valid UUID, by default False

    Returns
    -------
    str
        A valid UUID string, with no dashes (-).
        Either generate one, or uid if uid is a valid UUID.

    Raises
    ------
    ValueError
        If raise_invalid and uid is not a valid UUID.

    """
    try:
        uid = uuid.UUID(uid, version=1)
    except (ValueError, TypeError) as e:
        if raise_invalid:
            raise ValueError from e
        else:
            uid = uuid.uuid1()
    return str(uid).replace("-", "")


def purge_registry(sims=None):
    if sims is not None:
        for s in sims:
            if isinstance(s, Simulation):
                s = s.uid
            sim_registry.pop(s, None)
    else:
        sim_registry.clear()


def purge_caches(keys=None):
    for s in sim_registry.values():
        s.purge_cache(keys)


def get_sim(sim_or_uid):
    """Retreives a simulation from the register, building it if not already
    present.

    The eventual Simulation initialization uses default arguments (except for group, if
    provided).

    Raises
    ------
    ValueError, TypeError
        If sim_or_uid is neither null nor invalid.

    """
    # handle scalar array, e.g. from iterating over xarray.DataArray
    sim_or_uid = as_scalar(sim_or_uid)  # raises ValueError for non-scalar
    if isinstance(sim_or_uid, Simulation) or sim_or_uid is masked:
        return sim_or_uid  # OPT should we still insert in registry?
    # None and "" evaluate to False, but nan (xarray's masked) doesn't
    # check before initializing the Simulation, which may:
    # generate a dummy uid (None) or raise TypeError (nan)
    # NOTE isnan may raise TypeError, should we let Simulation() validate?
    if not sim_or_uid or not isinstance(sim_or_uid, str) and isnan(sim_or_uid):
        return
    if sim_or_uid not in sim_registry:
        sim_registry[sim_or_uid] = Simulation(sim_or_uid)
        logger.debug(f"Cached simulation {sim_or_uid}")
    return sim_registry[sim_or_uid]


def sim_like_arg(fun_sim):
    @wraps(fun_sim)
    def fun_sim_or_uid(sim, *args, **kwargs):
        return fun_sim(get_sim(sim), *args, **kwargs)

    return fun_sim_or_uid


def sims_iter_like_arg(func_sims=None, expand=False):
    if func_sims is None:
        return partial(sims_iter_like_arg, expand=expand)

    @wraps(func_sims)
    def func_cast(sims_like, *args, **kwargs):
        # TODO support sims_like collection (iterable? nah) of sims-like
        # (globs, iterable of uids)? note that we cannot use *sims_like
        if isinstance(sims_like, str):
            sims_like = SimsQuery(sims_like)
        sims = map(get_sim, sims_like)
        return func_sims([*sims] if expand else sims, *args, **kwargs)

    return func_cast


class Simulation(Cache):
    def __init__(self, uid=None, cfg=None, readonly=True):
        # init Cache & link rc I/O
        super().__init__(readonly=readonly)
        if uid and readonly:
            try:
                self.uid = uid.rsplit("~", 1)[0]
            except AttributeError as e:
                # otherwise we must catch AttrbiuteError in Measure.__call__
                msg = f"Expected string-like uid, got {type(uid).__name__}"
                raise TypeError(msg) from e
        else:
            self.uid = valid_uuid(uid)
        cfg = cfg or {}
        self.cfg_path = None
        self._save_time = None
        self._cpu_clock = time.process_time()
        self.cache = {}

        for key in rc["IO-handlers"]:
            if key != "dat":
                self.link(key)  # setattr as well?

        # setup logging
        if not readonly:
            self.setup_logging()
            logger.info("Running %s", shlex.join(sys.argv))

        # handle readonly uninitiazlized simulation
        try:
            par = self.load("par")
        except FileNotFoundError:
            par = {}
            if not readonly:
                raise
        if readonly and not par:
            self["par"] = cfg

        # merge config & runtime info into params
        if not readonly:
            cfg["uuid"] = self.uid
            cfg["versioning"] = {
                tag: run(
                    shlex.split(cmd),
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                for tag, cmd in rc["versioning"].items()
            }

            # update params
            diff = dictdiffer.diff(par, cfg, expand=True)
            diff = [d for d in diff if "remove" not in d]
            if diff:
                dictdiffer.patch(diff, par, in_place=True)
                msg = "\n".join(" ".join(str(v) for v in d) for d in diff)
                logger.warning("Config changes\n%s\n%s", msg, "=" * 80)

    @classmethod
    def from_config(cls, uid, group=None, template=None):
        # before writing/linking anything get config
        cfg_path, cfg = cfg_load(uid, group)
        sim = cls(uid, cfg, readonly=False)
        sim.cfg_path = cfg_path
        cfg_update_uid(cfg_path, uid, f"{sim.uid}~R", template)
        return sim

    def close(self):
        if not self.readonly and self.cfg_path:
            cfg_update_uid(self.cfg_path, f"{self.uid}~R", self.uid, template=False)

    # prevent numpy from iterating over self, but might be removed:
    # https://numpy.org/devdocs/reference/arrays.interface.html#object.__array_interface__
    # alternative: __len__ = None, but bool() breaks (and possibly other stuff as well)
    __array_interface__ = {"shape": (), "typestr": "O"}

    def __eq__(self, other):
        # implies self.uid == self, to distinguish use "is"
        return self.uid == getattr(other, "uid", other)

    def __lt__(self, other):
        return self.uid < getattr(other, "uid", other)

    def __repr__(self):
        args = f"{self.uid!r}, readonly={self.readonly!r}"
        return f"{type(self).__name__}({args}){set(self)}"

    def __str__(self):
        return self.uid

    def _repr_html_(self):
        path = self.handles["par"].storage
        return f'<tt><a href="{path}">{self.uid}</a></tt>'

    def __copy__(self):
        new = super().__copy__()
        new.uid = valid_uuid()
        new.cache = {}
        return new

    def copy(self, register=False):
        # enforce consistent copy() and __copy__()
        # UserDict built in implementations differ
        new = self.__copy__()
        if register:
            sim_registry[new.uid] = new
        return new

    def purge_cache(self, keys=None):
        if keys is not None:
            for k in keys:
                self.cache.pop(k, None)
        else:
            self.cache.clear()

    def runtime_info(self, ext_cpu_time=0.0):
        """Integrates simulation params with runtime info and returns it."""
        # TODO: ext_cpu_time ugly (used by extensions.ext_qtea)
        info = {}
        # cpu time
        cpu_time_path = "monitoring/cpu_time"
        delta = time.process_time() - self._cpu_clock
        self._cpu_clock += delta + ext_cpu_time
        cpu_time = dpath.get(self["par"], cpu_time_path, default=0.0)
        dpath.new(info, cpu_time_path, cpu_time + delta)

        self["par"] |= info
        return info

    def setup_logging(self):
        """Setup logging."""

        levels = rc["logging-levels"]
        format = rc["logging-format"]
        handlers = {}
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
        handlers["file"] = {
            "class": "logging.FileHandler",
            "filename": self.handles["log"].storage,
            # TODO: public access to storage
        }
        for h in handlers.values():
            h["formatter"] = "fmt"
        loggingrc = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"fmt": dict(format)},
            "handlers": handlers,
            "loggers": {k: {"level": l} for k, l in levels.items()},
            "root": {"handlers": list(handlers)},
        }  # yapf: disable
        logging.config.dictConfig(loggingrc)
        logging.captureWarnings(True)

    def link(self, key, **link_kw):
        # TODO:
        # if key in rc['IO-handlers']:
        #     raise ValueError(f'IO key {key} is reserved')

        handlers = rc["IO-handlers"]
        h = handlers.get(key) or handlers["dat"]
        h = Template(h).substitute(uid=self.uid, key=key)
        rc_link_kw = dict(
            zip(
                ("path", "write_mode", "serializer"),
                (s.strip() for s in h.split(",")),
            ),
        )
        # TODO: assert path is subpath of a rc directory
        # https://stackoverflow.com/questions/3812849/how-to-check-whether-a-directory-is-a-sub-directory-of-another-directory
        return super().link(key, **(rc_link_kw | link_kw))

    def dump(self, wait=0, **keyvals):
        if self._save_time and (time.monotonic() - self._save_time < wait):
            return False
        else:
            start_dump_time = time.process_time()
            self.runtime_info()
            super().dump(**(self.writable | keyvals))
            self._save_time = time.monotonic()
            logger.info(f"Dumped, took {time.process_time() - start_dump_time:.1f}s")
            return True
