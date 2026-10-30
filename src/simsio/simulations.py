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
 meta   | metadata | I/O | txt | simsio's own record: uuid, vcs, cpu_time, assets

A handler whose path template contains ``$key`` (e.g. ``dat``) declares an *asset
family* rather than a single storage: it is not linked at construction, but serves
as the template ``link`` draws on for extra keys. The ``assets`` section of ``meta``
records those links (handler name and serializer, never a resolved path) so that
reopening the simulation re-links them: `.simsiorc` owns *where* an asset lives, the
registry owns *how* it is encoded.

"""

import logging
import shlex
import sys
import time
import typing
import uuid
from cmath import isnan  # cmath just to be extra safe
from functools import partial, wraps
from pathlib import Path
from string import Template
from subprocess import run

import dictdiffer
from numpy.ma import masked  # numpy dependency :(

from simsio.configs import SimsQuery, cfg_claim, cfg_update_uid
from simsio.iocore import Cache, IOHandler
from simsio.settings import check_rc, rc
from simsio.utils import as_scalar, attr_name, get_module_attr, setup_logging

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

META = "meta"  # reserved IO key: simsio's own per-simulation record
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


def default_sim_class():
    """The Simulation class named by `[simulation] class` in the rc (a dotted path).

    Resolved on call, not at import: an extension subclass imports this module.
    """
    return get_module_attr(
        rc.get("simulation", "class", fallback="simsio.simulations.Simulation")
    )


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
    # None and "" evaluate to False, but nan (xarray's masked)
    # does not check before initializing the Simulation, which may:
    # generate a dummy uid (None) or raise TypeError (nan)
    # NOTE isnan may raise TypeError, should we let Simulation() validate?
    if not sim_like or not isinstance(sim_like, str) and isnan(sim_like):
        return
    if sim_like not in sim_registry:
        sim_registry[sim_like] = default_sim_class()(sim_like)
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
    save_extras = True
    """Whether `stash(..., extra=True)` writes; `run_sim` sets it from --save-extras."""

    def __init__(self, uid=None, cfg=None, readonly=True):
        """Readonly: opens `uid`, whatever it is (a missing one has `cfg` as par).

        Writable: `uid=None` is a new simulation with a fresh UUID; otherwise `uid` must
        be a UUID with a stored par, or a `cfg` (merged into it); else raises before
        anything is written.
        """
        check_rc()
        # init Cache & link rc I/O
        super().__init__(readonly=readonly)
        if not readonly and uid is not None and valid_uuid(uid) != uid:
            raise ValueError(f"{uid!r} is not a UUID: uid=None for a new simulation")
        if uid and readonly:
            try:
                self.uid = uid.rsplit("~", 1)[0]
            except AttributeError as e:
                # otherwise must catch AttrbiuteError in Measure.__call__
                msg = f"Expected string-like uid, got {type(uid).__name__}"
                raise TypeError(msg) from e
        else:
            self.uid = valid_uuid(uid)
        if not (readonly or uid is None or cfg is not None or self._par_stored()):
            msg = f"Simulation {self.uid} has neither a stored par nor a cfg"
            raise FileNotFoundError(msg)
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

        try:
            self.load(META)
        except FileNotFoundError:
            self[META] = {}

        if not readonly:
            self._merge("par", cfg)
            versions = {
                tag: run(
                    shlex.split(cmd),
                    capture_output=True,
                    text=True,
                    check=False,  # OPT check?
                ).stdout.strip()
                for tag, cmd in (rc["vcs"] if rc.has_section("vcs") else {}).items()
            }
            # empty output, e.g. `printenv` of an unset variable, records nothing
            vcs = {tag: v for tag, v in versions.items() if v}
            self._merge(META, {"uuid": self.uid, "vcs": vcs})

        # restore the assets registry last: a broken one must not stop par from loading
        self._restore_assets()

    def _merge(self, key, new):
        """Adds `new` to the stored `key`, logging the differences; removals are ignored."""
        old = self[key]
        diff = [d for d in dictdiffer.diff(old, new, expand=True) if "remove" not in d]
        if diff:
            dictdiffer.patch(diff, old, in_place=True)
            msg = "\n".join(" ".join(str(v) for v in d) for d in diff)
            # pure additions are the normal case (fresh sim, or a new option on a
            # re-run): only a changed value deserves to stand out in the log
            changed = any(d[0] != "add" for d in diff)
            level = logging.WARNING if changed else logging.INFO
            logger.log(level, "%s changes\n%s\n%s", key, msg, "=" * 80)

    def _restore_assets(self):
        # a re-run adds to the record, it does not replace it
        prev = self[META].setdefault("assets", {})
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
            # only a writable sim can act on it; a reader fails on access anyway
            level = logging.INFO if self.readonly else logging.WARNING
            logger.log(level, "Registered assets with no file: %s", ", ".join(missing))

    @property
    def assets(self):
        """Registry of the extra assets owned: {key: link spec}. Linked, not loaded."""
        return self.data[META].get("assets", {})

    @classmethod
    def from_config(cls, uid, group=None, template=None):
        # a name (n1) is a new simulation, a UUID key reruns that one
        new_uid = uid if valid_uuid(uid) == uid else valid_uuid()
        # claim the entry before writing anything: a concurrent claim raises KeyError
        cfg_path, cfg = cfg_claim(uid, f"{new_uid}~R", group, template)
        sim = cls(new_uid, cfg, readonly=False)
        sim.cfg_path = cfg_path
        return sim

    def close(self):
        if not self.readonly and self.cfg_path:
            try:
                cfg_update_uid(self.cfg_path, f"{self.uid}~R", self.uid, template=False)
            except KeyError as e:  # the entry was edited away meanwhile
                logger.warning(e.args[0])

    # prevent numpy from iterating over self, but might be removed:
    # https://numpy.org/devdocs/reference/arrays.interface.html#object.__array_interface__
    # alternative: __len__ = None, but bool() breaks (and possibly other stuff as well)
    __array_interface__: typing.ClassVar = {"shape": (), "typestr": "O"}

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
        # TODO cfg_path:line
        def entry(key):
            h = self.handles.get(key)  # a registered asset may have failed to link
            if h and h.serializer.typ == "t":  # only text is worth opening
                return f'<a href="{h.storage}">{key}</a>'
            return key

        storages = [k for k in rc["IO-handlers"] if k in self.handles]
        html = f"<kbd>{self.uid}</kbd> " + ", ".join(map(entry, storages))
        if tail := list(self.assets):  # not declared in the rc (potentially many)
            summary = f"{html} (+{len(tail)})"
            details = ", ".join(map(entry, tail))
            html = f"<details><summary>{summary}</summary>{details}</details>"
        return html

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
        """Accumulates this process's CPU time into `meta`."""
        # TODO: ext_cpu_time ugly (used by extensions.ext_qtea)
        delta = time.process_time() - self._cpu_clock
        self._cpu_clock += delta + ext_cpu_time
        meta = self[META]
        meta["cpu_time"] = meta.get("cpu_time", 0.0) + delta

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
        _, keyed = _handler(via)
        if key in handlers:
            if keyed:
                raise ValueError(f"{key!r} is an asset family, not a storage")
        elif not keyed:
            raise ValueError(f"handler {via!r} has no $key: cannot host {key!r}")
        elif Path(key).name != key:  # a separator would escape the family directory
            raise ValueError(f"asset key {key!r} must be a single path component")
        register = not self.readonly and key not in handlers
        # template-resolved paths cannot escape the rc directory (key is a single component)
        # explicit path= could, but only reached in readonly mode (_register rejects it)
        out = super().link(key, touch=touch, **(self._rc_link_kw(key, via) | link_kw))
        if register:
            self._register(key, {"via": via} | link_kw)
        return out

    def _rc_link_kw(self, key, via=None):
        """`path`, `write_mode`, `serializer` of `key` from the `via` rc handler."""
        h = Template(_handler(via or key)[0]).substitute(uid=self.uid, key=key)
        keys = ("path", "write_mode", "serializer")
        return dict(zip(keys, (s.strip() for s in h.split(","))))

    def _par_stored(self):
        kw = self._rc_link_kw("par")
        return self.storage("par", kw["path"], kw["serializer"])[0].is_file()

    def _register(self, key, spec):
        if "path" in spec:
            # the registry stores handler names so that a shared/moved tree still
            # resolves through the receiver's .simsiorc; a path would freeze the layout
            raise ValueError("path= cannot be registered, use via=")
        # record the serializer actually used, even when it came from the handler: the
        # registry owns encoding, so an asset stays readable if that default changes
        spec |= {"serializer": attr_name(self.handles[key].serializer)}
        # bypass __getitem__/setdefault: both would attempt a load
        self.data[META]["assets"][key] = spec
        logger.debug("Registered asset %s via %s", key, spec["via"])

    def unlink(self, key):
        self.assets.pop(key, None)
        return super().unlink(key)

    def stash(self, key, val, extra=False, **link_kw):
        """Dumps one asset right away and drops it from the cache, keeping it linked.

        Unlike `dump`, which rewrites every cached writable handle, this writes `key`
        alone -- so stashing in a loop stays linear -- together with the registry, so
        that a crash cannot leave an unregistered file behind.

        An `extra` (bulk, optional) asset is skipped, neither linked nor registered,
        unless `save_extras`. Returns the storage, or None if skipped.
        """
        if extra and not self.save_extras:
            logger.debug("Extra %r not saved (see --save-extras)", key)
            return None
        if key not in self.handles or link_kw:
            self.link(key, **link_kw)
        self[key] = val
        IOHandler.dump(self, **{key: val, META: self[META]})
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
