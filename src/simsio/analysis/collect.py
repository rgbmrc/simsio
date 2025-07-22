import re
import logging
from functools import cached_property, wraps

import numpy as np

from simsio.settings import rc
from simsio.simulations import get_sim, UID_DTYPE
from simsio.configs import yamlsf, path_to_group, glob_groups
from simsio.analysis.quantitites import Measure

logger = logging.getLogger(__name__)


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
    """Sorts a set of uids in lexicographic order according to the values of the given
    parmeters."""
    keys, vals = _get_params_vals(sims, keys)
    idxs = np.lexsort(vals[::-1])
    sims = list(sims)  # need __getitem__
    sims = [sims[i] for i in idxs]
    if return_vals:
        vals = tuple(zip(*vals))
        vals = [vals[i] for i in idxs]
        return sims, vals
    return sims
