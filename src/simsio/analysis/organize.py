import logging
from collections.abc import Hashable, Sequence
from warnings import warn

import numpy as np
import xarray as xr

from simsio.analysis.quantities import Function, Measure
from simsio.configs import SimsQuery
from simsio.simulations import Simulation, get_sim, get_sims_iter, sims_iter_like_arg

__all__ = ["get_sims_array", "stack_grids", "uids_grid", "uids_sort"]

logger = logging.getLogger(__name__)
np.set_printoptions(formatter={"object": str})


get_sims_array = np.frompyfunc(get_sim, 1, 1)


def _get_sims_attrs(sims, keys):
    return {k: k(sims) for k in map(Measure.get, keys)}


def _discard_erroring(sims, keys):
    """Drop sims that raise while computing any of `keys`."""
    bad = set()
    for sim in sims:
        try:
            for k in keys:
                k(sim)
        except ValueError:
            bad.add(sim.uid)
    if bad:
        _uids = ", ".join(bad)
        warn(f"discarded {len(bad)} sims erroring on grid keys: {_uids}")
        sims = np.fromiter((s for s in sims if s.uid not in bad), object)
    return sims


def uids_grid(sims, keys) -> xr.DataArray:
    keys = [*map(Measure.get, keys)]
    match sims:
        case str():
            group = sims
        case SimsQuery():
            group = sims.group_globs[0]
        case _:
            group = None
    sims = np.fromiter(get_sims_iter(sims), object)  # does not iterate over sim dict
    sims = _discard_erroring(sims, keys)
    inds = np.empty((len(keys), len(sims)), dtype=np.intp)
    coords = []
    for j, vs in enumerate(_get_sims_attrs(sims, keys).values()):
        _coords, inds[j] = np.unique(vs, return_inverse=True)
        coords.append(_coords)
    # could call unique on vals directly (as for lexsort in uids_sort)
    # but we need coords anyway and then unique is faster on inds
    uniq, jj, inv, counts = np.unique(inds, True, True, True, axis=1)
    for u in np.flatnonzero(counts > 1):
        bad = np.flatnonzero(inv == u)[1:]
        _vals = ", ".join(str(coords[ik][iv]) for ik, iv in enumerate(uniq[:, u]))
        _uids = ", ".join(sims[i].uid for i in bad)
        warn(f"discarded {len(bad)} sims with duplicate coords ({_vals}): {_uids}")
    shape = tuple(map(len, coords))
    grid = np.empty(shape, dtype=object)
    for i, j in zip(uniq.T, jj):
        grid[*i] = sims[j]  # fancy indexing on grid would trigger copy
    # xarray works best with string names
    # https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-name
    # we can also keep the (hashable) Measure objects as duplicate coords
    # here explicit tuple coercion is required for non-string names
    # https://github.com/pydata/xarray/issues/2292#issuecomment-2341989713
    # coords = {k.name: v for k, v in uniq.items()}
    # coords |= {k: (k.name, v) for k, v in uniq.items()}
    return xr.DataArray(grid, coords, tuple(k.name for k in keys), group)


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
    prepend: Sequence[str | Hashable] | None = None,
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


def _dim_names(obs):
    """The grid-dimension names among `obs`, an arbitrarily nested structure of
    quantities. Anything that is not a named quantity is skipped."""
    names = []
    for o in np.ravel(Function.get_array(obs)):
        try:
            names.append(Function.get(o).name)
        except (TypeError, AttributeError):  # not a quantity, or None
            pass
    return names


def transpose_grid(
    grid: xr.DataArray, dims: list[Hashable] | dict[str, Hashable], reserve=()
):
    """Give each slot of `dims` a grid dimension and transpose the grid onto them.

    A slot asks for a dimension by name (a Function, Measure or string), for the next
    unclaimed one (`...`), or for none at all (None or False, a size-1 dummy). Unclaimed
    dimensions trail the slots. Returns the transposed grid and one quantity per slot,
    to title it with.

    `reserve` holds the quantities drawn *inside* a tile (x/y observables): the
    dimensions they name are withheld from the slots and pinned as the trailing axes,
    in the order given, so that they collapse into a single tile instead of being
    spread over the grid. Quantities that do not name a dimension are ignored.

    """
    # TODO FunctionLike in type hints?
    try:
        slots = list(dims.items())
    except AttributeError:
        slots = [(f"dim_{i}", o) for i, o in enumerate(dims)]
    free = list(grid.dims)
    # withheld before any slot can claim them, dummy slots taking their place below
    reserve = list(dict.fromkeys(d for d in _dim_names(reserve) if d in free))
    for dim in reserve:
        free.remove(dim)
    axes, titles = {}, {}

    def claim(k, dim):
        free.remove(dim)
        axes[k] = dim

    for k, o in slots:  # skipped: the dim nest_grids made for the slot, or a dummy
        if not o:  # None or False
            titles[k] = None
            if k in free:
                claim(k, k)
            else:
                axes[k] = k  # expanded below
    for k, o in slots:  # named, before a positional claim can steal what they ask for
        if k in axes or o is ...:
            continue
        titles[k] = o
        try:
            # DEL .name if Function coords is implemented
            dim = Function.get(o).name
        except TypeError:  # not a dimension: filled below, keeping the title
            continue
        if dim in reserve:
            raise ValueError(f"{dim!r} is both tile content and the {k!r} slot")
        if dim in free:
            claim(k, dim)
    for k, _ in slots:  # the rest, in grid order, then dummies once they run out
        if k in axes:
            continue
        if free:
            axes[k] = dim = free.pop(0)
            # HACK if to avoid casting to Function "row" etc
            titles.setdefault(k, dim if dim in grid.coords else None)
        else:
            axes[k] = k
            titles.setdefault(k, None)

    order = [axes[k] for k, _ in slots]
    grid = grid.expand_dims([dim for dim in order if dim not in grid.dims])
    return (
        grid.transpose(*order, ..., *reserve),
        [Function.get(titles[k]) for k, _ in slots],
    )
