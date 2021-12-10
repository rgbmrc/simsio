"""
**Abbreviations**
- sim: simulation
- uid: simulation identifier (human provided key or computer generated uuid)


**Storage elements** associated with a Simulation instance

I/O | typ | name | fullname | description
----------------------------------------------------------------------
  O | txt | log  | log      | cronological summary of the simulation
  O | bin | res  | results  | measurements to be tabulated or plotted
I/O | txt | par  | params   | options used by routines and classes
I/O | bin | dat  | data     | raw output, i.e. not par, res or log
I   | txt | cfg  | config   | subset of par provided as user input
"""

import fcntl
import logging
import os
import signal
import sys
import time
import uuid
from collections import deque
from contextlib import contextmanager
from functools import cached_property, wraps
from itertools import product
from pathlib import Path
from string import Template
from subprocess import CalledProcessError, run

import dictdiffer
import dpath.util as dpath
import h5py
import numpy as np
import ruamel.yaml as yaml
import tenpy
from tenpy.algorithms.truncation import TruncationError
from tenpy.tools.events import EventHandler
from tenpy.tools.hdf5_io import load_from_hdf5, save_to_hdf5
from tenpy.tools.misc import setup_logging
from tenpy.tools.params import Config, asConfig
from tenpy.tools.process import mkl_set_nthreads, omp_set_nthreads

__version__ = 21.12

logger = logging.getLogger(__name__)

yamlrt = yaml.YAML(typ='rt')
yamlsf = yaml.YAML(typ='safe')
yamlsf.default_flow_style = False
yamlsf.representer.ignore_aliases = lambda *args: True
yamlsf_register = yamlsf.representer.add_multi_representer
yamlsf_register(np.integer, lambda dumper, data: dumper.represent_int(data))
yamlsf_register(np.floating, lambda dumper, data: dumper.represent_float(data))
yamlsf_register(
    Config,
    lambda dumper, data: dumper.represent_dict(data.as_dict()),
)
yamlsf_register(
    TruncationError,
    lambda dumper,
    data: dumper.represent_dict(data.__dict__),
)


def _valid_uuid(uid=None, raise_invalid=False):
    """
    Returns and/or check the validity of a UUID (universally unique
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
    return str(uid).replace('-', '')


LOGGER_LEVELS = {'tenpy.tools.params': 'WARNING'}
MERGE_TAG = '$$$'
MERGE_REF = '<<<'
CFG_DIR = 'configs/'
OUT_DIR = 'out/'
CFG_EXT = '.yaml'
PAR_EXT = '.yaml'
RES_EXT = '.npz'
DAT_EXT = '.hdf5'
LOG_EXT = '.log'
PARAMS_LOCK_ATTEMPT_NUM = 30
PARAMS_LOCK_ATTEMPT_FREQ = 1
UID_DTYPE = np.array(_valid_uuid()).dtype

cache = {}

_config_path_history = deque([], maxlen=100)

shelve = False
checkpoint = EventHandler()


def set_shelve(signum, frame):
    global shelve
    shelve = True
    logger.warning(f'Signal {signum} received: shelving at checkpoint')


def build_measures(measures, **context):
    # TODO: handle special notation for expectation values & correlation functions?
    return {
        k: m if callable(m) else eval(m, context)
        for k, m in measures.items()
    }


def checkpointed(iterable):
    for i in iterable:
        logger.info('checkpoint after iteration step completed')
        checkpoint.emit()
        yield i


def append_measures(measures, results, target=None):
    results |= {k: list(results.get(k, [])) for k in measures}
    samples, = {len(results[k]) for k in measures}
    while not target or samples < target:
        if samples:
            yield samples
        for k, msr in measures.items():
            try:
                res = np.asanyarray(msr())
            except:
                logger.error(f'Unable to measure {k}')
                raise
            results[k].append(res)
        samples += 1


@contextmanager
def run_simulation(**sim_kwargs):
    # TODO: generalize (or delegate) args parsing
    try:
        signal.signal(signal.SIGUSR1, set_shelve)
        group, uid, nthreads = sys.argv[1:]
        nthreads = int(nthreads)
        omp_set_nthreads(nthreads) or mkl_set_nthreads(nthreads)
        sim_kwargs.setdefault('cache', False)
        sim_kwargs.setdefault('readonly', False)
        sim = Simulation(uid, group, **sim_kwargs)
        checkpoint.connect(sim.dump)
    except:
        logger.exception('Uncaught exception while loading simulation')
        raise

    try:
        yield sim
    except:
        logger.exception('Uncaught exception while running simulation')
        raise
    finally:
        t = time.process_time()
        logger.info('starting dump')
        sim.dump()
        logger.info(f'finished dump after: {time.process_time() - t:.0f}s')
        sim.params.warn_unused(recursive=True)


def configs_path(glob=None, configs_dir=CFG_DIR, configs_ext=CFG_EXT):
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
    glob = (glob or '*') + configs_ext
    paths = set(Path(configs_dir).glob(glob))
    return (
        [p for p in _config_path_history if p in paths] +
        [p for p in paths if not p in _config_path_history]
    )


def get_uids(glob=None):
    """
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
    configs_root = (set(yamlsf.load(p)) for p in configs_path(glob))
    return set.union(*configs_root) - {MERGE_TAG}


def sim_to_uid_arg(fun_sim):

    @wraps(fun_sim)
    def fun_uid(uid, *args, **kwargs):
        return fun_sim(get_sim(uid), *args, **kwargs)

    return fun_uid


@sim_to_uid_arg
def extract_log(sim, pattern, reverse=False, op='search'):
    log = sim.log
    if reverse:
        log = '\n'.join(reversed(log.splitlines()))
    return getattr(pattern, op)(log)


def _extract_dict(dict, glob, op):
    return getattr(dpath, op)(dict, glob)


@sim_to_uid_arg
def extract_param(sim, glob, op='get'):
    return _extract_dict(sim.params, glob, op)


@sim_to_uid_arg
def extract_result(sim, glob, op='get'):
    return _extract_dict(sim.results, glob, op)


def gen_configs(template, params, glob=None):
    generated = {}
    for path in configs_path(glob):
        configs = yamlrt.load(path)
        template = Template(configs[MERGE_TAG][template])
        uids = generated[path.stem] = set()
        for vals in product(*params.values()):
            yml = template.substitute(dict(zip(params.keys(), vals)))
            c = yamlrt.load(yml)
            configs |= c
            uids |= set(c)
        yamlrt.dump(configs, path)
    return generated


def get_sim(uid):
    """
    Retreives a simulation from the module cache, building it if not
    already present.

    Parameters
    ----------
    uid : str
        The uid of the simulation.

    Returns
    -------
    Simulation
        The simulation.
    """
    return cache.get(uid) or Simulation(uid)


def _get_params_vals(uids, keys):
    pars = (get_sim(uid).params for uid in uids)
    for i, k in enumerate(keys):
        if isinstance(k, str):
            keys[i] = (k, dpath._DEFAULT_SENTINAL)
    vals = [[dpath.get(p, k, default=d) for k, d in keys] for p in pars]
    return list(zip(*vals))


def uids_sort(uids, keys):
    """
    Sorts a set of uids in lexicographic order according to the values
    of the given parmeters.

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
    vals = _get_params_vals(uids, keys)
    idxs = np.lexsort(reversed(vals))
    return [uids[i] for i in idxs]


def uids_grid(uids, keys):
    # TODO: aliases for paths
    vals = _get_params_vals(uids, keys)
    idxs = np.empty((len(keys), len(uids)), dtype=np.intp)
    uniq = {}
    for j, ((k, d), v) in enumerate(zip(keys, vals)):
        u, i = np.unique(v, return_inverse=True)
        uniq[k] = u
        idxs[j] = i
    idxs = idxs.T
    grid = np.empty([len(u) for u in uniq.values()], dtype=UID_DTYPE)
    for i, uid in zip(idxs, uids):
        grid[tuple(i)] = uid

    return grid, uniq


def pop(*uids, group=None):
    cfgs_paths = {}
    for u in uids:
        p, _ = load_config(u, group, expand=False)
        cfgs_paths.setdefault(p, set()).add(u)
    for p, us in cfgs_paths.items():
        with lock_config(p) as cfg:
            for u in us:
                cfg.pop(u)
                sim_p = sim_path(u)
                for ext in (
                    LOG_EXT,
                    RES_EXT,
                    PAR_EXT,
                    DAT_EXT,
                ):
                    sim_p.with_suffix(ext).unlink(True)
                    logger.info(f'Deleted {u}')


def _merge(dst, src):
    if isinstance(dst, dict) and isinstance(src, dict):
        for k in src:
            if k not in dst:
                dst[k] = src[k]
            else:
                _merge(dst[k], src[k])


def _expand(config, templates):
    if isinstance(config, dict):
        refs = config.pop(MERGE_REF, [])
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
    for path in configs_path(group):
        cfgs = yamlsf.load(path)
        if cfg := cfgs.get(uid):
            logger.info(f'Config for {uid} found in {path}')
            while path in _config_path_history:
                _config_path_history.remove(path)
            _config_path_history.appendleft(path)
            break
    else:
        raise ValueError(f'Simulation {uid} config not found')

    if expand:  # expand config via templates
        _expand(cfg, cfgs.get(MERGE_TAG, {}))

    return path, cfg


@contextmanager
def lock_config(path):
    with open(path, 'r+') as f:
        for _ in range(PARAMS_LOCK_ATTEMPT_NUM):
            try:
                logger.debug(f'Attempting to acquire lock on {path}')
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                error = e
                time.sleep(1.0 / PARAMS_LOCK_ATTEMPT_FREQ)
                continue
            else:
                cfgs = yamlrt.load(f)
                yield cfgs
                f.seek(0)
                yamlrt.dump(cfgs, f)
                break
        else:
            raise error


def sim_path(uid) -> Path:
    return Path(OUT_DIR, uid)


class Simulation:

    def __init__(self, uid, group=None, cache=True, readonly=True):
        _uid, uid = uid, uid if readonly else _valid_uuid(uid)
        self.uid = uid
        self.cpu_time = time.process_time()
        self.savetime = time.monotonic()
        self.readonly = readonly
        self.path = sim_path(uid)

        if not readonly:
            # create output directory
            self.path.parent.mkdir(parents=True, exist_ok=True)

        # setup logging
        log_file = None if readonly else self.path.with_suffix(LOG_EXT)
        setup_logging(filename=log_file, logger_levels=LOGGER_LEVELS)

        # load params file (if present)
        try:
            params = yamlsf.load(self.path.with_suffix(PAR_EXT))
        except FileNotFoundError:
            params = None

        # load config
        # when readonly, proceed only if no params file is present
        if not readonly or not params:
            path, _params = load_config(_uid, group)

            # update uid in config if needed
            if uid != _uid:
                with lock_config(path) as cfg:
                    cfg.insert(list(cfg).index(_uid), uid, cfg.pop(_uid))
                logger.info(f'Assigned uid {uid}')

            # integrate config with dynamically generated info
            try:
                git_rev = run(
                    ['git', 'rev-parse', 'HEAD'],
                    capture_output=True,
                    text=True,
                )
                git_rev = git_rev.stdout.strip()
            except CalledProcessError:
                git_rev = None
            dpath.new(_params, 'versioning/tenpy_verison', tenpy.__version__)
            dpath.new(_params, 'versioning/git_revision', git_rev)
            if not params:
                dpath.new(_params, 'monitoring/cpu_time', 0.0)

            # update params using config
            params = params or {}
            diff = [
                d for d in dictdiffer.diff(params, _params, expand=True)
                if not 'remove' in d
            ]
            if diff:
                dictdiffer.patch(diff, params, in_place=True)
                diff = '\n'.join(' '.join(str(v) for v in d) for d in diff)
                logger.warning('\n'.join(('Config changes', diff, '=' * 80)))

        self.params = asConfig(params, self.__class__.__name__)
        self.params.touch('versioning', 'monitoring')

        if cache:
            cache[uid] = self
            logger.debug(f'Cached simulation {uid}')

    # use cached_property for lazy loading of results and allow for `del sim.results`
    @cached_property
    def results(self):
        try:
            with np.load(self.path.with_suffix(RES_EXT)) as f:
                results = dict(f)
        except FileNotFoundError:
            results = {}
        return results

    # use cached_property to allow for `del sim.data`
    @cached_property
    def data(self):
        return {}

    @property
    def log(self):
        return self.path.with_suffix(LOG_EXT).read_text()

    def load_data(
        self,
        subpath='/',
        ignore_unknown=False,
        exclude=None,
        cache=True,
    ):
        # A dict subclass overriding __getitem__ to load from file is not an option.
        # Reloading would require a `del data[key]` first.
        # Moreover, consider a hdf5 file with the following structure
        # /groupA
        #    /dataset1
        #    /dataset2
        # and suppose you read /groupA/dataset1 and then /groupA; the key "groupA"
        # already exists, thus with a naive `key in self` check you end up either
        # reading two times "dataset1" or missing "dataset2".
        # With this function the choice of the behaviour is left to the user:
        # load_data(key) implements reloading & the first, data[key] the second.
        try:
            with h5py.File(self.path.with_suffix(DAT_EXT), 'r') as f:
                d = load_from_hdf5(f, subpath, ignore_unknown, exclude)
            logger.info(f'Loaded data/{subpath}')
        except (FileNotFoundError, KeyError) as e:
            raise KeyError(f'Invalid path: data/{subpath}') from e
        else:
            if cache:
                subpath = subpath.strip('/')
                if subpath:
                    dpath.new(self.data, subpath, d)
                else:
                    self.data = d
        return d

    def dump(self, wait=0):
        if self.readonly:
            raise PermissionError('Simulation is readonly')
        if self.savetime and (time.monotonic() - self.savetime < wait):
            return False

        # TODO: delegate all params updates to function
        cpu_time = time.process_time()
        self.params['monitoring']['cpu_time'] += cpu_time - self.cpu_time
        self.cpu_time = cpu_time

        # make sure the output directory exists
        os.makedirs(self.path.parent, exist_ok=True)

        def bak_path(p):
            bak = '_bak'
            try:
                return p + bak  # p string
            except TypeError:
                return p.with_suffix(p.suffix + bak)  # p pathlib.Path

        saved = []  # store saved objects
        try:  # bakup last saving and save
            saving = 'data'
            f = self.path.with_suffix(DAT_EXT)
            with h5py.File(f, 'a') as f:
                for k in self.data:
                    saving = f'data/{k}'
                    if k in f:
                        f.move(k, bak_path(k))
                    save_to_hdf5(f, self.data[k], k)
                    saved.append(saving)
                    logger.debug(f'Saved {saving}')

            saving = 'results'
            f = self.path.with_suffix(RES_EXT)
            if f.is_file():
                os.replace(f, bak_path(f))
            np.savez(f, **self.results)
            saved.append(saving)
            logger.debug(f'Saved {saving}')

            saving = 'params'
            f = self.path.with_suffix(PAR_EXT)
            if f.is_file():
                os.replace(f, bak_path(f))
            yamlsf.dump(self.params, f)
            saved.append(saving)
            logger.debug(f'Saved {saving}')

            self.savetime = time.monotonic()

        except:
            logger.error(f'Could not save {saving}; already saved: {saved}')
            raise

        else:  # cleanup backup
            with h5py.File(self.path.with_suffix(DAT_EXT), 'a') as f:
                for k in self.data:
                    k_bak = bak_path(k)
                    if k_bak in f:
                        del f[k_bak]
            for ext in (RES_EXT, PAR_EXT):
                f_bak = bak_path(self.path.with_suffix(ext))
                if f_bak.is_file():
                    os.remove(f_bak)

            return True
