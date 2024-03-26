"""
**Abbreviations**
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

import fcntl
import logging
import logging.config
import re
import shlex
import sys
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import cached_property, wraps
from itertools import chain, product
from pathlib import Path
from shutil import rmtree
from string import Template
from subprocess import run

import dictdiffer
import dpath.util as dpath
import numpy as np
import ruamel.yaml as yaml

from simsio import rc
from simsio.iocore import Cache

logger = logging.getLogger(__name__)

yamlsf = yaml.YAML(typ="safe")
yamlrt = yaml.YAML(typ="rt")
yamlrt.width = 8192


def _valid_uuid(uid=None, raise_invalid=False):
    """
    Returns and/or check the validity of a UUID (universally unique identifier).

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


UID_DTYPE = np.array(_valid_uuid()).dtype
UID_REGEX = "[a-z0-9]{32}"

CFG_EXT = ".yaml"
CFG_DIR = Path(rc["configs"]["directory"])
CFG_LOCK_ATTEMPT_FREQ = 1

# TODO: use file cache for _config_path_history
HISTORY_FILE = ".simsio_history"
_config_path_history = deque(maxlen=100)


def sim_or_uid_arg(fun_sim):
    @wraps(fun_sim)
    def fun_sim_or_uid(sim, *args, **kwargs):
        return fun_sim(get_sim(sim), *args, **kwargs)

    return fun_sim_or_uid


@sim_or_uid_arg
def extract_text(sim, key, regex, reverse=False, op="search"):
    d = sim[key]
    if reverse:
        d = "\n".join(reversed(d.splitlines()))
    return getattr(re.compile(regex), op)(d)


@sim_or_uid_arg
def extract_dict(sim, key, glob, op=None):
    op = op or dpath.get
    return op(sim[key], glob)


def glob_groups(pattern=None, cron=False):
    """
    Returns the paths of config files matching a glob.

    Parameters
    ----------
    group : str, optional
        Glob pattern to match (extension excluded), by default '*
    configs_dir : str, optional
        Directory where to look for config files, by default CONFIGS_DIR
    configs_ext : str, optional
        Extension of config files, by default CONFIGS_EXT

    Returns
    -------
    list[Path]
        Paths of matching config files, ordered chronologically, from the most recently used process
    """
    pattern = (pattern or "**/*") + CFG_EXT
    paths = CFG_DIR.glob(pattern)
    if cron:
        paths = set(paths)
        paths = chain(
            (p for p in _config_path_history if p in paths),
            (p for p in paths if not p in _config_path_history),
        )
    return paths


def get_uids(glob=None):
    """
    DEPRECATED, use sims_query.

    Returns uids contained in config files matching a glob.

    Parameters
    ----------
    group : str, optional
        Glob pattern the config filename should match, by default '*'

    Returns
    -------
    set[str]
        uids in matching configs.
    """
    root = (set(yamlsf.load(p)) for p in glob_groups(glob))
    return set.union(*root) - {rc["configs"]["header_tag"]}


def path_to_group(p):
    return str(p.relative_to(CFG_DIR).with_suffix(""))


def group_to_path(g):
    return Path(CFG_DIR, g).with_suffix(CFG_EXT)


class SimsQuery:
    def __init__(self, group_glob=None, uid_regex=UID_REGEX):
        self.group_glob = group_glob
        self.uid_regex = uid_regex

        tag = rc["configs"]["header_tag"]
        if not uid_regex:
            uid_filter = lambda u: u != tag
        else:
            uid_regex = re.compile(uid_regex, re.S)
            uid_filter = lambda u: u != tag and uid_regex.fullmatch(u)

        self.groups = {
            path_to_group(p): {u for u in yamlsf.load(p) if uid_filter(u)}
            for p in glob_groups(group_glob)
        }  # yapf: disable

    @cached_property
    def uids(self):
        return {u: g for g, us in self.groups.items() for u in us}

    def __repr__(self):
        cls = self.__class__.__name__
        args = f"group_glob={self.group_glob!r}, uid_regex={self.uid_regex!r}"
        return f"{cls}({args})"


def gen_configs(template, params, glob=None):
    generated = {}
    for path in glob_groups(glob):
        configs = yamlrt.load(path)
        header = configs[rc["configs"]["header_tag"]]
        template = Template(header[template])
        uids = generated[path.stem] = set()
        try:
            keys = params.keys()
            vals = params.values()
        except AttributeError:
            pass
        else:
            params = [dict(zip(keys, vs)) for vs in product(*vals)]
        for enum, ps in enumerate(params):
            yml = template.substitute(ps, enum=str(enum))
            c = yamlrt.load(yml)
            configs |= c
            uids |= set(c)
        yamlrt.dump(configs, path)
    return generated


def get_sim(sim_or_uid, group=None):
    """
    Retreives a simulation from the cache, building it if not already present.

    The eventual Simulation initialization uses default arguments
    (except for group, if provided).
    """
    if not sim_or_uid:
        return None
    if isinstance(sim_or_uid, Simulation):
        return sim_or_uid
    if sim_or_uid not in Simulation.cache:
        Simulation.cache[sim_or_uid] = Simulation(sim_or_uid, group)
        logger.debug(f"Cached simulation {sim_or_uid}")
    return Simulation.cache[sim_or_uid]


def _get_params_vals(sims, keys):
    try:
        sims = sims.items()
    except AttributeError:
        sims = ((s,) for s in sims)
    pars = (get_sim(*sim).par for sim in sims)
    for i, k in enumerate(keys):
        if isinstance(k, str):
            keys[i] = (k, dpath._DEFAULT_SENTINEL)
    vals = [[dpath.get(p, k, default=d) for k, d in keys] for p in pars]
    return tuple(zip(*vals))


def uids_sort(sims, keys, return_vals=False):
    """
    Sorts a set of uids in lexicographic order according to the values of the given
    parmeters.

    Parameters
    ----------
    uids : [type]
        [description]
    keys : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    sims = list(sims)
    vals = _get_params_vals(sims, keys)
    idxs = np.lexsort(vals[::-1])
    sims = [sims[i] for i in idxs]
    if return_vals:
        vals = tuple(zip(*vals))
        vals = [vals[i] for i in idxs]
        return sims, vals
    return sims


def uids_grid(sims, keys):
    # TODO: aliases for paths
    vals = _get_params_vals(sims, keys)
    idxs = np.empty((len(keys), len(sims)), dtype=np.intp)
    uniq = {}
    for j, ((k, d), v) in enumerate(zip(keys, vals)):
        u, i = np.unique(v, return_inverse=True)
        uniq[k] = u
        idxs[j] = i
    idxs = idxs.T
    grid = np.empty([len(u) for u in uniq.values()], dtype=UID_DTYPE)
    # grid = np.ma.masked_all([len(u) for u in uniq.values()], dtype=UID_DTYPE, fill_value='')
    for i, s in zip(idxs, sims):
        grid[tuple(i)] = getattr(s, "uid", s)
    return grid, uniq


def sort_config(glob, keys):
    tag = rc["configs"]["header_tag"]
    for path in glob_groups(glob):
        with lock_config(path) as cfg:
            header = cfg.pop(tag, None)
            for u in reversed(uids_sort(cfg, keys)):
                cfg.insert(0, u, cfg.pop(u))
            if header:
                cfg.insert(0, tag, header)


def pop(*uids, group=None):
    cfgs_paths = defaultdict(set)
    for u in uids:
        p, _ = load_config(u, group, expand=False)
        cfgs_paths[p].add(u)
    for p, us in cfgs_paths.items():
        with lock_config(p) as cfg:
            for u in us:
                for h in rc["IO-handlers"].values():
                    glob = h.split(",")[0].strip()
                    glob = Template(glob).substitute(uid=u, key="*")
                    for p in Path(dir).glob(u):  # TODO: make recursive?
                        if p.is_file():
                            p.unlink()
                        elif p.is_dir():
                            rmtree(p)
                cfg.pop(u)
                logger.info(f"Deleted {u}")


def _merge(dst, src):
    if isinstance(dst, dict) and isinstance(src, dict):
        for k in src:
            if k not in dst:
                dst[k] = src[k]
            else:
                _merge(dst[k], src[k])


def _expand(config, templates):
    if isinstance(config, dict):
        refs = config.pop(rc["configs"]["header_ref"], [])
        if isinstance(refs, str):
            refs = [refs]
        for k in config:
            _expand(config[k], templates)
        for r in reversed(refs):
            _merge(config, templates[r])
    elif isinstance(config, list):
        for v in config:
            _expand(v, templates)


def load_config(uid, group=None, expand=True):
    if uid in (rc["configs"]["header_tag"], rc["configs"]["header_ref"]):
        raise KeyError(f"Key {uid} is reserved")
    for path in glob_groups(group):
        cfgs = yamlsf.load(path)
        if cfgs and (cfg := cfgs.get(uid)):
            while path in _config_path_history:
                _config_path_history.remove(path)
            _config_path_history.appendleft(path)
            break
    else:
        raise KeyError(f"Simulation {uid} config not found")

    if expand:  # expand config via templates
        _expand(cfg, cfgs.get(rc["configs"]["header_tag"], {}))

    return path, cfg


@contextmanager
def lock_config(path):
    with open(path, "r+") as f:
        for _ in range(rc["configs"].getint("lock_attempts")):
            try:
                logger.debug(f"Attempting to acquire lock on {path}")
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                error = e
                time.sleep(1.0 / CFG_LOCK_ATTEMPT_FREQ)
                continue
            else:
                cfgs = yamlrt.load(f)
                yield cfgs
                f.seek(0)
                f.truncate()
                yamlrt.dump(cfgs, f)
                break
        else:
            raise error


class Simulation(Cache):

    cache = {}

    def __init__(self, uid, group=None, readonly=True):
        # init Cache & link rc I/O
        super().__init__(readonly=readonly)

        self.uid = uid.rsplit("-", 1)[0] if readonly else _valid_uuid(uid)
        self._save_time = None
        self._cpu_clock = time.process_time()
        self.cfg_path = None
        cfg = {}

        # before writing/linking anything get config
        if not readonly:
            self.cfg_path, cfg = load_config(uid, group)
            # update uid in config
            with lock_config(self.cfg_path) as cfgs:
                cfgs.insert(list(cfgs).index(uid), f"{self.uid}-R", cfgs.pop(uid))

        for key in rc["IO-handlers"]:
            if key != "dat":
                self.link(key)

        # setup logging
        if not readonly:
            self.setup_logging()
            logger.info(
                f"Running {shlex.join(sys.argv)}, config found in {self.cfg_path}",
            )

        # handle readonly uninitiazlized simulation
        try:
            par = self.load("par")
        except FileNotFoundError:
            par = None
            if not readonly:
                raise
        if readonly and not par:
            _, self["par"] = load_config(uid, group)

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
            diff = dictdiffer.diff(self["par"], cfg, expand=True)
            diff = [d for d in diff if not "remove" in d]
            if diff:
                dictdiffer.patch(diff, self["par"], in_place=True)
                msg = "\n".join(" ".join(str(v) for v in d) for d in diff)
                msg = "\n".join(("Config changes", msg, "=" * 80))
                logger.warning(msg)

        # self['par'].touch('version', 'monitor')

    def close(self):
        if not self.readonly and self.cfg_path:
            uid_R = f"{self.uid}-R"
            # update uid in config
            with lock_config(self.cfg_path) as cfgs:
                cfgs.insert(list(cfgs).index(uid_R), self.uid, cfgs.pop(uid_R))

    def __repr__(self):
        cls = self.__class__.__name__
        args = f"{self.uid!r}, readonly={self.readonly!r}"
        return f"{cls}({args}){set(self)}"

    def __getattribute__(self, name):
        if name in rc["IO-handlers"]:
            return self[name]
        else:
            return super().__getattribute__(name)

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
