import re
import logging
from functools import cached_property, wraps

import numpy as np
import dpath

from ..settings import rc
from ..simulations import Simulation, UID_DTYPE
from ..configs import yamlsf, path_to_group, glob_groups
from .quantitites import Measure

logger = logging.getLogger(__name__)


sim_registry = {}


def purge_registry(sims=None):
    if sims is not None:
        for s in np.ravel(sims):
            if isinstance(s, Simulation):
                s = s.uid
            sim_registry.pop(s, None)
    else:
        sim_registry.clear()


def purge_caches(keys=None):
    for s in sim_registry.values():
        s.purge_cache(keys)


def get_sim(sim_or_uid, group=None):
    """
    Retreives a simulation from the register, building it if not already present.

    The eventual Simulation initialization uses default arguments
    (except for group, if provided).
    """
    if isinstance(sim_or_uid, Simulation) or sim_or_uid is np.ma.masked:
        return sim_or_uid
    if not isinstance(sim_or_uid, str):
        raise TypeError(f"Expected str uid, got {type(sim_or_uid).__name__}")
    if not sim_or_uid:
        return np.ma.masked
    if sim_or_uid not in sim_registry:
        sim_registry[sim_or_uid] = Simulation(sim_or_uid, group)
        logger.debug(f"Cached simulation {sim_or_uid}")
    return sim_registry[sim_or_uid]


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


class SimsQuery:
    def __init__(self, *group_globs, valid_uuid=True, select=None):
        self.group_globs = group_globs or ["**/*"]
        self.valid_uuid = valid_uuid
        if self.valid_uuid:
            # hardcoded default for backward compatibility with old .simsiorc files
            # TODO remove when default rc file is deployed
            uuid_regex = rc["configs"].get("uuid_regex", "[a-z0-9]{32}")
            uid_filter = re.compile(uuid_regex, re.S).fullmatch
        else:
            uid_filter = rc["configs"]["header_tag"].__ne__
        self.groups = {
            path_to_group(p): set(filter(uid_filter, cfg))
            for glob in self.group_globs
            for p in glob_groups(glob)
            if (cfg := yamlsf.load(p))  # skip non-iterable empty yaml (=None)
        }
        if select is not None:
            self.groups = {k: set(filter(select, us)) for k, us in self.groups.items()}

    @cached_property
    def uids(self):
        return {u: g for g, us in self.groups.items() for u in us}

    def __iter__(self):
        return iter(self.uids)

    def __len__(self):
        return len(self.uids)

    def __repr__(self):
        args = f"group_globs={self.group_globs}, valid_uuid={self.valid_uuid}"
        return f"{type(self).__name__}({args})"


def _get_params_vals(sims, keys):
    try:
        sims = sims.items()
    except AttributeError:
        sims = ((s,) for s in sims)
    sims = (get_sim(*sim) for sim in sims)
    keys = [Measure.get(k) for k in keys]
    vals = [[k(sim) for k in keys] for sim in sims]
    return keys, tuple(zip(*vals))


def sims_or_group_arg(func_sims):
    @wraps(func_sims)
    def func_sims_or_group(sims_or_group, *args, **kwargs):
        if isinstance(sims_or_group, str):
            sims_or_group = SimsQuery(sims_or_group)
        return func_sims(sims_or_group, *args, **kwargs)

    return func_sims_or_group


@sims_or_group_arg
def uids_grid(sims, keys) -> tuple[np.ndarray, dict[Measure, np.ndarray]]:
    # NOTE why returning array of strings and not Simulation objects?
    # among other reasons:
    # https://github.com/numpy/numpy/issues/27212#issue-2465378354
    keys, vals = _get_params_vals(sims, keys)
    idxs = np.empty((len(keys), len(sims)), dtype=np.intp)
    uniq = {}
    for j, (k, v) in enumerate(zip(keys, vals)):
        u, i = np.unique(v, return_inverse=True)
        uniq[k] = u
        idxs[j] = i
    shape = tuple(map(len, uniq.values()))
    grid = np.ma.masked_all(shape, dtype=UID_DTYPE)
    for i, s in zip(idxs.T, sims):
        grid[tuple(i)] = getattr(s, "uid", s)
    if not grid.mask.any():
        grid = grid.data
    return grid, uniq


@sims_or_group_arg
def uids_sort(sims, keys, return_vals=False):
    """
    Sorts a set of uids in lexicographic order according to the values of the given
    parmeters.
    """
    keys, vals = _get_params_vals(sims, keys)
    idxs = np.lexsort(vals[::-1])
    sims = list(sims)  # need __getitem__
    sims = [sims[i] for i in idxs]
    if return_vals:
        vals = tuple(zip(*vals))
        vals = [vals[i] for i in idxs]
        return sims, vals
    return sims
