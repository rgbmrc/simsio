"""**Abbreviations**

 - sim: simulation
 - uid: simulation identifier (human provided key or computer generated uuid)

**Storage elements** associated with a Simulation instance

 name | fullname | I/O | typ | description
 ---------------------------------------------------------------------
 log    | log      |   O | txt | cronological summary of the simulation
 res    | results  |   O | bin | measurements to be tabulated or plotted
 par    | params   | I/O | txt | options used by routines and classes
 dat    | data     | I/O | bin | any persistent object, all the above included
 cfg    | config   | I   | txt | subset of par provided as user input
 assets | assets   | I/O | txt | registry of the extra assets this sim owns

A handler whose path template contains ``$key`` (e.g. ``dat``) declares an *asset
family* rather than a single storage: it is not linked at construction, but serves
as the template ``link`` draws on for extra keys. The ``assets`` registry records
those links (handler name and serializer, never a resolved path) so that reopening
the simulation re-links them: `.simsiorc` owns *where* an asset lives, the registry
owns *how* it is encoded.

"""

import logging
import shlex
import sys
import time
import uuid
from cmath import isnan  # cmath just to be extra safe
from functools import partial, wraps
from pathlib import Path
from string import Template
from subprocess import run

import dictdiffer
import dpath
from numpy.ma import masked  # numpy dependency :(

from simsio.configs import SimsQuery, cfg_load, cfg_update_uid
from simsio.iocore import Cache, IOHandler
from simsio.settings import rc
from simsio.utils import as_scalar, attr_name, setup_logging

__all__ = [
    "Simulation",
    "get_sim",
    "purge_caches",
    "purge_registry",
    "sim_like_arg",
    "sim_registry",
    "sims_iter_like_arg",
    "valid_uuid",
]

logger = logging.getLogger(__name__)

ASSETS = "assets"  # reserved IO key: the per-simulation asset registry
sim_registry: dict[str, "Simulation"] = {}


def _handler(key):
    """Handler entry for `key`, and whether it is an asset family ($key template)."""
    tpl = rc["IO-handlers"][key]
    return tpl, "key" in Template(tpl).get_identifiers()


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


def get_sim(sim_like):
    """Retreives a simulation from the register, building it if not already
    present.

    The eventual Simulation initialization uses default arguments.

    Raises
    ------
    ValueError, TypeError
        If sim_or_uid is neither null nor invalid.

    """
    # handle scalar array, e.g. from iterating over xarray.DataArray
    sim_like = as_scalar(sim_like)  # raises ValueError for non-scalar
    if isinstance(sim_like, Simulation) or sim_like is masked:
        return sim_like  # OPT should we still insert in registry?
    # None and "" evaluate to False, but nan (xarray's masked) doesn't
    # check before initializing the Simulation, which may:
    # generate a dummy uid (None) or raise TypeError (nan)
    # NOTE isnan may raise TypeError, should we let Simulation() validate?
    if not sim_like or not isinstance(sim_like, str) and isnan(sim_like):
        return
    if sim_like not in sim_registry:
        sim_registry[sim_like] = Simulation(sim_like)
        logger.debug("Cached simulation %s", sim_like)
    return sim_registry[sim_like]


def get_sims_iter(sims_like):
    # TODO support sims_like collection (iterable? nah) of sims-like
    # (globs, iterable of uids)? note that we cannot use *sims_like
    if isinstance(sims_like, str):
        sims_like = SimsQuery(sims_like)
    return map(get_sim, sims_like)


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
        sims = get_sims_iter(sims_like)
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
            if not _handler(key)[1]:  # keyed templates are families, linked on demand
                self.link(key)  # setattr as well?

        # setup logging
        if not readonly:
            setup_logging(self.handles["log"].storage)
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

        # restore the assets registry last: a broken one must not stop par from loading
        if ASSETS in self.handles:
            self._restore_assets()

    def _restore_assets(self):
        try:
            prev = self.load(ASSETS, cache=False) or {}
        except FileNotFoundError:
            prev = {}
        self[ASSETS] = prev  # a re-run adds to the record, it does not replace it
        missing = []
        for key, spec in list(prev.items()):  # link re-registers, i.e. writes into prev
            try:
                # touch=False: a registered asset may legitimately be absent, e.g. when
                # only results/ was shared; an empty file would then mask it as EOFError
                self.link(key, touch=False, **spec)
            except Exception:  # a missing serializer must not make the sim unopenable
                logger.exception("Cannot link registered asset %s", key)
            else:
                self.handles[key].storage.is_file() or missing.append(key)
        if missing:
            logger.warning("Registered assets with no file: %s", ", ".join(missing))

    @property
    def assets(self):
        """Registry of the extra assets owned: {key: link spec}. Linked, not loaded."""
        return self.data.get(ASSETS, {})

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
        # TODO cfg_path:line, dyanimic keys
        paths = (self.handles["par"].storage, self.handles["log"].storage)
        links = map('[<a href="{}">{}</a>]'.format, paths, ("par", "log"))
        return "<tt>" + "".join((self.uid, *links)) + "</tt>"

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
        if keys is None:
            self.cache.clear()
        else:
            for k in keys:
                self.cache.pop(k, None)

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

    def link(self, key, via=None, touch=True, **link_kw):
        """Links `key`, resolving it through the `via` handler if it is not one itself.

        `via` defaults to the key's own handler, else to "dat"; it must name an asset
        family (a $key template). An asset linked on a writable simulation is recorded
        in the assets registry, so re-linking one on reopening is idempotent.
        """
        handlers = rc["IO-handlers"]
        if via and key in handlers:
            raise ValueError(f"IO key {key!r} is reserved")
        via = via or (key if key in handlers else "dat")
        tpl, keyed = _handler(via)
        if key in handlers:
            if keyed:
                raise ValueError(f"{key!r} is an asset family, not a storage")
        elif not keyed:
            raise ValueError(f"handler {via!r} has no $key: cannot host {key!r}")
        elif Path(key).name != key:  # a separator would escape the family directory
            raise ValueError(f"asset key {key!r} must be a single path component")
        register = not self.readonly and ASSETS in self.data and key not in handlers
        h = Template(tpl).substitute(uid=self.uid, key=key)
        rc_link_kw = dict(
            zip(
                ("path", "write_mode", "serializer"),
                (s.strip() for s in h.split(",")),
            ),
        )
        # template-resolved paths cannot escape the rc directory (key is a single component)
        # explicit path= could, but only reached in readonly mode (_register rejects it)
        out = super().link(key, touch=touch, **(rc_link_kw | link_kw))
        if register:
            self._register(key, {"via": via} | link_kw)
        return out

    def _register(self, key, spec):
        if "path" in spec:
            # the registry stores handler names so that a shared/moved tree still
            # resolves through the receiver's .simsiorc; a path would freeze the layout
            raise ValueError("path= cannot be registered, use via=")
        # record the serializer actually used, even when it came from the handler: the
        # registry owns encoding, so an asset stays readable if that default changes
        spec |= {"serializer": attr_name(self.handles[key].serializer)}
        # bypass __getitem__/setdefault: both would attempt a load
        self.data[ASSETS][key] = spec
        logger.debug("Registered asset %s via %s", key, spec["via"])

    def unlink(self, key):
        self.assets.pop(key, None)
        return super().unlink(key)

    def stash(self, key, val, **link_kw):
        """Dumps one asset right away and drops it from the cache, keeping it linked.

        Unlike `dump`, which rewrites every cached writable handle, this writes `key`
        alone -- so stashing in a loop stays linear -- together with the registry, so
        that a crash cannot leave an unregistered file behind.
        """
        if key not in self.handles or link_kw:
            self.link(key, **link_kw)
        self[key] = val
        keyvals = {key: val}
        if ASSETS in self.handles:
            keyvals[ASSETS] = self.assets
        IOHandler.dump(self, **keyvals)
        del self[key]
        return self.handles[key].storage

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
