import logging
from collections.abc import Sequence
from warnings import warn
from typing import Hashable

import numpy as np
import xarray as xr

from simsio.analysis.quantities import Function, Measure
from simsio.configs import SimsQuery
from simsio.simulations import get_sim, get_sims_iter, sims_iter_like_arg, Simulation

__all__ = ["get_sims_array", "uids_grid", "uids_sort", "stack_grids"]

logger = logging.getLogger(__name__)
np.set_printoptions(formatter={"object": str})


get_sims_array = np.frompyfunc(get_sim, 1, 1)


def _get_sims_attrs(sims, keys):
    return {k: k(sims) for k in map(Measure.get, keys)}


def uids_grid(sims, keys) -> xr.DataArray:
    match sims:
        case str():
            group = sims
        case SimsQuery():
            group = sims.group_globs[0]
        case _:
            group = None
    sims = np.fromiter(get_sims_iter(sims), object)  # does not iterate over sim dict
    inds = np.empty((len(keys), len(sims)), dtype=np.intp)
    coords = {}
    for j, (k, vs) in enumerate(_get_sims_attrs(sims, keys).items()):
        coords[k], inds[j] = np.unique(vs, return_inverse=True)
    # could call unique on vals directly (as for lexsort in uids_sort)
    # but we need coords anyway and then unique is faster on inds
    inds, js, counts = np.unique(inds, axis=1, return_index=True, return_counts=True)
    if duplicate := np.sum(counts - 1):
        warn(f"discarded {duplicate} simulations with duplicate coords")
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
    return xr.DataArray(grid, coords.values(), tuple(k.name for k in coords), group)


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


# TODO subclass instead?
# https://docs.xarray.dev/en/stable/internals/extending-xarray.html
# https://github.com/pydata/xarray/issues/3980
# but UID grids only require limited functionality
@xr.register_dataarray_accessor("simsio")
class UIDSGrid:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj: xr.DataArray = xarray_obj

    def add_dim(self, obs, axis) -> xr.DataArray:
        obs = Measure.get(obs)
        val = np.unique(obs(self._obj))
        assert val.size == 1
        return self._obj.expand_dims({obs.name: val}, axis)

    def transpose(self, *dim: Hashable, **transpose_kwds):
        names = [Function.get(d).name if d is not ... else d for d in dim]
        return self._obj.transpose(*names, **transpose_kwds)


def stack_grids(grids, axis=0, dim=None, **concat_kwds):
    if axis not in {0, -1}:
        raise ValueError("axis must be 0 or -1")
    existing_dims = {d for gr in grids for d in gr.dims}
    if not dim:
        i, dim = 0, "stack"
        while dim in existing_dims:
            i += 1
            dim = f"stack_{i}"
    else:
        assert dim not in existing_dims
    grid = xr.concat(grids, dim, **concat_kwds)
    if axis == -1:
        grid = grid.transpose(..., dim)
    return grid.reset_coords(drop=True)


def _nested_grid_depth(grid):
    if isinstance(grid, xr.DataArray):
        return 0
    if isinstance(grid, (tuple, list)):  # otherwise infinite recursion from str uids
        return 1 + max(_nested_grid_depth(x) for x in grid)
    raise TypeError


def nest_grids(
    nested: list,
    prepend: Sequence[str | Hashable] = None,
    concat_dim: str | Hashable = None,
    **combine_kwargs,
) -> xr.DataArray:
    """Stacks a nested list of DataArrays.

    Parameters
    ----------
        nested
            Arbitrarily nested list of xarray.DataArray objects.
        prepend
            Leading dimensions to insert; once exhausted use concat_dim.
        concat_dim
            After prepend, new dimensions will be `{concat_dim}_{i}`
            where `i` is the dimension index. Default: "grid_dim".
        **combine_kwargs
            Keyword args for xarray.combine_nested (excluding concat_dim).

    Returns
    -------
    A single xarray.DataArray with new leading dimensions.

    """
    concat_dim = concat_dim or "grid_dim"
    ndim = _nested_grid_depth(nested)
    dims = list(prepend or [])[:ndim]
    dims.extend(f"{concat_dim}_{i}" for i in range(len(dims), ndim))
    return xr.combine_nested(nested, dims, **combine_kwargs).transpose(*dims, ...)
