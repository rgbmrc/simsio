import logging
from itertools import starmap

import numpy as np
import xarray as xr

from simsio.configs import sims_or_group_arg
from simsio.analysis.quantitites import Measure
from simsio.simulations import get_sim

__all__ = ["uids_grid", "uids_sort"]

logger = logging.getLogger(__name__)


def get_params_vals(sims, keys):
    try:
        sims = sims.items()
    except AttributeError:
        sims = ((s,) for s in sims)
    keys = [Measure.get(k) for k in keys]
    vals = [[k(sim) for k in keys] for sim in starmap(get_sim, sims)]
    return keys, tuple(zip(*vals))


@sims_or_group_arg
def uids_grid(sims, keys) -> xr.DataArray:
    # NOTE why returning array of strings and not Simulation objects?
    # among other reasons:
    # https://github.com/numpy/numpy/issues/27212#issue-2465378354
    keys, vals = get_params_vals(sims, keys)
    idxs = np.empty((len(keys), len(sims)), dtype=np.intp)
    uniq = {}
    for j, (k, v) in enumerate(zip(keys, vals)):
        u, i = np.unique(v, return_inverse=True)
        uniq[k] = u
        idxs[j] = i
    shape = tuple(map(len, uniq.values()))
    grid = np.empty(shape, dtype=object)
    for i, s in zip(idxs.T, sims):
        grid[tuple(i)] = getattr(s, "uid", s)
    # xarray works best with string names
    # https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-name
    # we can also keep the (hashable) Measure objects as duplicate coords
    # here explicit tuple coercion is required for non-string names
    # https://github.com/pydata/xarray/issues/2292#issuecomment-2341989713
    # coords = {k.name: v for k, v in uniq.items()}
    # coords |= {k: (k.name, v) for k, v in uniq.items()}
    return xr.DataArray(grid, uniq.values(), tuple(k.name for k in keys))


@sims_or_group_arg
def uids_sort(sims, keys, return_vals=False):
    """Sorts a set of uids in lexicographic order according to the values of the given
    parmeters."""
    keys, vals = get_params_vals(sims, keys)
    idxs = np.lexsort(vals[::-1])
    sims = list(sims)  # need __getitem__
    sims = [sims[i] for i in idxs]
    if return_vals:
        vals = tuple(zip(*vals))
        vals = [vals[i] for i in idxs]
        return sims, vals
    return sims


def mask_grid(grid, cond, drop=False):
    # register a "simsio" accessor instead?
    # https://docs.xarray.dev/en/stable/internals/extending-xarray.html
    return grid.where(cond, "", drop=drop)


@xr.register_dataarray_accessor("simsio")
class UIDSGrid:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj: xr.DataArray = xarray_obj

    def add_dim(self, obs, axis) -> xr.DataArray:
        obs = Measure.get(obs)
        val = np.unique(obs(self._obj))
        assert val.size == 1
        return self._obj.expand_dims({obs.name: val}, axis)
