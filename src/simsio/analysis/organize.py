import logging
from warnings import warn

import numpy as np
import xarray as xr

from simsio.analysis.quantitites import Measure
from simsio.simulations import get_sim, sims_iter_like_arg, Simulation

__all__ = ["uids_grid", "uids_sort"]

logger = logging.getLogger(__name__)
np.set_printoptions(formatter={"object": str})


get_sims_array = np.frompyfunc(get_sim, 1, 1)


def _get_sims_attrs(sims, keys):
    return {k: k(sims) for k in map(Measure.get, keys)}


@sims_iter_like_arg
def uids_grid(sims, keys) -> xr.DataArray:
    sims = np.fromiter(sims, object)  # does not iterate over sim dict
    inds = np.empty((len(keys), len(sims)), dtype=np.intp)
    coords = {}
    for j, (k, vs) in enumerate(_get_sims_attrs(sims, keys).items()):
        coords[k], inds[j] = np.unique(vs, return_inverse=True)
    # could call unique on vals directly (as for lexsort in uids_sort)
    # but we need coords anyway and then unique is faster on inds
    inds, js, counts = np.unique(inds, axis=1, return_index=True, return_counts=True)
    if dupl := np.sum(counts - 1):
        warn(f"discarded {dupl} simulations with duplicate coords")
    shape = tuple(map(len, coords.values()))
    grid = np.empty(shape, dtype=object)
    for i, j in zip(inds.T, js):
        grid[*i] = sims[j]  # fancy indexing on grid would trigger copy
    # xarray works best with string names
    # https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-name
    # we can also keep the (hashable) Measure objects as duplicate coords
    # here explicit tuple coercion is required for non-string names
    # https://github.com/pydata/xarray/issues/2292#issuecomment-2341989713
    # coords = {k.name: v for k, v in uniq.items()}
    # coords |= {k: (k.name, v) for k, v in uniq.items()}
    return xr.DataArray(grid, coords.values(), tuple(k.name for k in keys))


@sims_iter_like_arg
def uids_sort(
    sims, keys, return_vals=False
) -> list[Simulation] | tuple[list[Simulation], list[tuple]]:
    """Sorts a set of uids in lexicographic order according to the values of
    the given parmeters."""
    sims = np.fromiter(sims, object)  # avoid array creation for every measure
    params = _get_sims_attrs(sims, keys)
    vals = tuple(params.values())
    inds = np.lexsort(vals[::-1])
    sims = sims[inds].tolist()  # return list, more flexible
    if return_vals:
        vals = tuple(zip(*vals))
        vals = [vals[i] for i in inds]
        return sims, vals
    return sims


@xr.register_dataarray_accessor("simsio")
class UIDSGrid:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj: xr.DataArray = xarray_obj

    def add_dim(self, obs, axis) -> xr.DataArray:
        obs = Measure.get(obs)
        val = np.unique(obs(self._obj))
        assert val.size == 1
        return self._obj.expand_dims({obs.name: val}, axis)
