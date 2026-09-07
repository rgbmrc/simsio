import logging
from copy import copy, deepcopy
from itertools import filterfalse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors, rcParams, ticker, transforms

from simsio.analysis.axes_grid import (
    OPPOSITE,
    SIDES,
    AxesGrid,
    overhangs,
    parse_sides,
    tile_overhangs,
)
from simsio.analysis.grids import LinearGrid, UniformGrid, bin_edges
from simsio.analysis.organize import nest_grids, transpose_grid
from simsio.analysis.quantities import Function, Measure
from simsio.analysis.utils import sanitize_path

__all__ = [
    "add_cbar",
    "annotate_image_axis",
    "apply_obs_props",
    "autoscale_norms",
    "axes_1d",
    "axes_2d",
    "grid_titles",
    "join_obs_names",
    "plot_1d_data",
    "plot_2d_data",
    "report_1d",
    "report_2d",
    "sanitize_fig_name",
    "sm_from_obs",
]

logger = logging.getLogger(__name__)

# TODO move to mplotter/config
TILE_SIZE = 1.33
AXES_PAD = 0.1
CBAR_SIZE = 0.1
MAX_DIGITIZED = 12

REPORT_1D_DIMS = ["row", "col", "cycler", "cmap"]
REPORT_2D_DIMS = ["row", "col"]


def fix_mfc(l):  # TODO move to mplotter
    c = l.get_color()
    l.set(mfc=colors.to_rgba(c, 0.2), mec=colors.to_rgba(c, 1.0))
    return l


# on top of matplotlib ones
_PROP_ALIASES = {"major_locator": "loc", "major_formatter": "fmt"}


def _parse_aliases(kwds):
    for name, alias in _PROP_ALIASES.items():
        if alias in kwds:
            if name in kwds:
                raise ValueError(f"Specified both '{name}' and '{alias}'")
            kwds[name] = kwds.pop(alias)
    return kwds


def _parse_obs_props(obs, props_name, **props):
    props = getattr(obs, props_name, {}) | props
    # props = parse_aliases(props) # NOTE enable if needed
    return props


def apply_obs_props(obj, obs, props_name, **props):
    props = _parse_obs_props(obs, props_name, **props)
    if props:  # avoid printing allowed values
        plt.setp(obj, **props)
    return props


def plot_1d_data(x_obs, y_obs, u=None, ax=None, **plot_kwds):
    ax = ax or plt.gca()
    apply_obs_props(ax.xaxis, x_obs, "axis_kwds")
    apply_obs_props(ax.yaxis, y_obs, "axis_kwds")
    with np.printoptions(threshold=4, edgeitems=2):
        try:
            y = y_obs(u)
            if x_obs:
                x = x_obs(u)
        except ValueError as e:
            logger.error(e)
            return None
    # use apply_obs_props on l instead?
    plot_kwds = _parse_obs_props(y_obs, "plot_kwds", **plot_kwds)
    if x_obs:
        l = ax.plot(x, y, **plot_kwds)
    else:
        l = ax.plot(y, **plot_kwds)
    return l


def axes_1d(ug, it, cbar_obs=None, plotting_func=None, plot_kwds=None, ax=None):
    plotting_func = plotting_func or plot_1d_data
    if cbar_obs:
        ax.sm = cbar_obs.sm
    for _, x_obs, y_obs in it:
        x_obs = x_obs.item()
        y_obs = y_obs.item()
        if y_obs is None:
            # no "off", there might be other obs in same axes
            continue
        _ug = ug[it.multi_index]
        _plot_kwds = plot_kwds.copy()
        if cbar_obs:
            try:
                z = np.ma.asanyarray(cbar_obs(_ug))
                z = np.unique(z[~z.mask]).item()
            except ValueError:
                logger.warning("No unique z-value from cbar_obs")
            else:
                _plot_kwds["color"] = cbar_obs.sm.to_rgba(z)
        _plot_kwds["label"] = y_obs
        if x_obs:
            ax.set_xscale(getattr(x_obs, "scale", "linear"))
            ax.set_xlabel(x_obs)
        ax.set_yscale(getattr(y_obs, "scale", "linear"))
        ax.set_ylabel(y_obs)
        # plotting function gets the final say on the axes
        plotting_func(x_obs, y_obs, _ug, ax=ax, **_plot_kwds)


def axes_2d(
    us, obs, x_obs=None, y_obs=None, plotting_func=None, plot_kwds=None, ax=None
):
    plotting_func = plotting_func or plot_2d_data
    return plotting_func(obs, us, x_obs=x_obs, y_obs=y_obs, ax=ax, **(plot_kwds or {}))


def report_1d(
    ug,
    x_obs,
    y_obs,
    cbar_obs=None,
    y_titles=...,
    x_titles=...,
    *,
    plotting_func=None,
    axes_func=None,
    tile_size=None,
    plot_kwds=None,
    label_side=None,
    title_side=None,
    fig=None,
    **grid_kwds,
):
    # DEL cycler=None hack while cycler not implemented
    dims = dict(zip(REPORT_1D_DIMS, [y_titles, x_titles, None, cbar_obs]))
    # TODO x/y order, cbar
    xg, br_arrays, titles = _report_dims(ug, dims, [x_obs, y_obs])
    ug, cbar_obs = br_arrays[0], titles[3]  # cbar_obs may have been a grid dim (...)
    _prepare_grid_kwds(grid_kwds, cbar_mode="single" if cbar_obs else None)
    fig, grid, sides = _report_grid(
        xg,
        ug.shape[:2],
        titles,
        [y_obs, x_obs, cbar_obs],
        tile_size,
        fig,
        grid_kwds,
        label_side,
        title_side,
    )
    # one cbar_obs per colorbar, each scaled over the tiles its bar serves
    cbar_obs = cbar_obs and _cbar_obs_grid(cbar_obs, ug, grid_kwds)

    axes_func = axes_func or axes_1d
    plot_kwds = deepcopy(plot_kwds) or {}
    grid_iter, plot_iter = np.nested_iters(
        br_arrays, [[0, 1], [2, 3]], ["refs_ok", "multi_index"], order="C"
    )
    for ax, _ in zip(grid, grid_iter):
        ij = grid_iter.multi_index
        obs = cbar_obs[ij] if cbar_obs is not None else None
        axes_func(ug[ij], plot_iter, obs, plotting_func, plot_kwds, ax)
        if ug.shape[2] > 1:
            ax.legend()  # TODO cycler
        if not ax.lines:
            ax.axis("off")
            if grid_kwds["cbar_mode"] == "each":
                ax.cax.set_visible(False)  # no data, no colorbar
        elif obs:
            add_cbar(ax, obs.sm, obs)
    _report_titles(fig, grid, ug, titles, sides, tile_size)
    return fig, grid


def plot_2d_data(obs, u=None, x_obs=None, y_obs=None, ax=None, **im_kwds):
    ax = ax or plt.gca()
    obs = Function.get(obs)
    x_obs = Function.get(x_obs)
    y_obs = Function.get(y_obs)
    if not obs:
        ax.axis("off")
        return None
    im_kwds = getattr(obs, "im_kwds", {}) | im_kwds
    # prevent autoscale of the original observable's norm
    # copies still needed also in grid e.g. when cbar_mode="each"
    # FIXME I am afraid imshow does not cover 2d plots with non-linear coords
    # https://stackoverflow.com/a/20844584
    # im_kwds.setdefault("extent", (*ext_from_obs(x_obs, u), *ext_from_obs(y_obs, u.T)))
    im_kwds.setdefault("cmap", copy(getattr(obs, "cmap", None)))
    im_kwds.setdefault("norm", copy(getattr(obs, "norm", None)))
    try:
        dat = obs(u)
    except ValueError:
        dat = np.ma.masked
    # first check because np.ma.masked_invalid raises TypeError
    # on np.ma.masked, from previous line but possibly also from obs(u)
    if dat is np.ma.masked or np.ma.masked_invalid(dat).mask.all():
        ax.text(0.5, 0.5, "no\nvalid\ndata", ha="center", va="center")
        # # NOTE disables titles as well, workaround:
        # https://stackoverflow.com/a/49155982/12379192
        # ax.axis("off")
        return None
    im = ax.imshow(dat, **im_kwds)
    add_cbar(ax, im, obs)
    annotate_image_axis(ax.xaxis, x_obs, u)
    annotate_image_axis(ax.yaxis, y_obs, u)
    return im


def grid_from_obs(obs, u):
    raise NotImplementedError  # TODO just a sketch
    vals = obs(u)
    uniq = uniq_flat_mask(obs(u))
    scale = getattr(obs, "scale", None)
    try:
        return UniformGrid.from_points(uniq, scale).extent
    except ValueError:
        return (-0.5, len(vals) - 0.5)


def _prepare_uids_grid(ug, dims):
    try:
        xg = nest_grids(ug, dims.keys(), "dat")
    except TypeError:
        xg = None
        dims = dims.values()
        ug = np.expand_dims(ug, [i for i, dim in enumerate(dims) if not dim])
    else:
        # OPT modify dims in place instead?
        xg, dims = transpose_grid(xg, dims)
        ug = xg.to_masked_array(copy=False)
    return xg, ug, dims


def _prepare_arrays(ndims, it_arrays):
    dummy = np.empty([1] * ndims)  # ensure at least ndims
    # transpose to broadcast from the leading dimension
    _, *br_arrays = np.broadcast_arrays(dummy, *map(np.transpose, it_arrays))
    return [*map(np.transpose, br_arrays)]


def _prepare_grid_kwds(grid_kwds, **defaults):
    """Fill in the grid defaults, in place."""
    # both pads are the *white space*: what the tiles draw into them is measured on
    # top of it, later, by fit_axes_pad
    defaults = {
        "axes_pad": AXES_PAD,
        "cbar_pad": AXES_PAD,
        "cbar_size": CBAR_SIZE,
    } | defaults
    for k, v in defaults.items():
        grid_kwds.setdefault(k, v)


def _sides(grid_kwds, titled, label_side=None, title_side=None):
    """Which sides of the grid carry the row/column titles and which the labels.

    Titles take the top and left edges, unless the colorbar is there; the labels take
    the opposite edge of a titled direction, or the matplotlib default -- but never a
    side that a per-tile colorbar owns. Both are given as (x, y). An explicit
    `title_side` overrides the first rule, colorbar or not: the bars then move out to
    leave the titles the room nearest the tiles.

    """
    mode = grid_kwds.get("cbar_mode")
    loc = grid_kwds.get("cbar_location", "right") if mode else None
    titles = tuple(OPPOSITE[s] if s == loc else s for s in ("top", "left"))
    if title_side is not None:
        titles = parse_sides(title_side, titles)
    if label_side is not None:
        return titles, parse_sides(label_side)
    labels = []
    for side, title, has_title in zip(("bottom", "left"), titles, titled):
        side = OPPOSITE[title] if has_title else side
        if mode == "each" and side == loc:  # every tile's own bar sits there
            side = OPPOSITE[side]
        labels.append(side)
    return titles, tuple(labels)


def _edge_axes(grid, side):
    """The tiles along the `side` edge of the grid."""
    return {
        "top": grid.axes_row[0],
        "bottom": grid.axes_row[-1],
        "left": grid.axes_column[0],
        "right": grid.axes_column[-1],
    }[side]


def _decorations_pad(axs, pos, past_cbar=True):
    """Points to clear whatever `axs` already draw on their `pos` side, their own
    colorbar included unless `past_cbar` is off -- then the pad stops at the tile, and
    what is placed there lands between it and its bar. Shared by the whole row/column,
    so that what is placed past it stays aligned."""
    over = max(
        (tile_overhangs if past_cbar else overhangs)(ax)[SIDES.index(pos)] for ax in axs
    )
    return 72 * over  # ScaledTranslation and labelpad work in points


def _report_dims(ug, dims, arrays):
    """Give the report slots their grid dimensions, then broadcast onto them: the uid
    grid first, then `arrays`. Comes before the figure, whose name needs the obs the
    slots resolved to."""
    xg, ug, titles = _prepare_uids_grid(ug, dims)
    return xg, _prepare_arrays(len(dims), [ug, *arrays]), titles


def _report_grid(
    xg, shape, titles, label_obs, tile_size, fig, grid_kwds, label_side, title_side
):
    """The figure and the axes to plot the grid in, sided and labelled."""
    titled = [_titled(titles[1]), _titled(titles[0])]  # x from the columns, y the rows
    sides = _sides(grid_kwds, titled, label_side, title_side)
    fig = plt.figure(
        fig or _fig_name(xg, label_obs),
        _fig_size(shape, tile_size, grid_kwds["axes_pad"]),
    )
    grid = AxesGrid(fig, 111, shape, label_side=sides[1], **grid_kwds)
    return fig, grid, sides


def _report_titles(fig, grid, ug, titles, sides, tile_size, obs=None, has_cbar=()):
    """Fit the pads to what the tiles ended up drawing, keeping the grid's axes_pad as
    the white space on top of it, then title the rows and the columns past those
    decorations -- or, on the side the colorbars are on, between the tiles and the
    bars, which are pushed out again to clear the titles."""
    loc = grid.cbar_location if grid.cbar_mode else None
    grown = np.zeros(2)
    fitted = grid.needs_pad_fit() or loc in sides[0]
    if fitted:
        grown = grid.fit_axes_pad()
    has_cbar = has_cbar or (None, None)
    for side, title, cbar in zip(sides[0], reversed(titles[:2]), has_cbar):
        grid_titles(_edge_axes(grid, side), side, ug, title, obs, cbar, side != loc)
    if fitted and loc in sides[0]:  # the fit came before the titles it must now clear
        grown[SIDES.index(loc) % 2] += grid.fit_cbar_pad()
    if fitted:
        size = _fig_size(grid.get_geometry(), tile_size, grid.get_axes_pad())
        # the figure never accounted for the colorbar: at least do not shrink the tiles
        fig.set_size_inches(size + grown)
    fig.align_labels()


def _fig_size(shape, tile_size, pad):
    # just an estimate: it ignores whatever the grid draws outside its tiles
    pad = np.asarray(pad, float)  # (horizontal, vertical)
    return np.flip(shape[:2]) * ((tile_size or TILE_SIZE) + pad)


def _fig_name(xg, label_obs):
    return getattr(xg, "name", "") + "/" + join_obs_names(*label_obs)


def report_2d(
    ug,
    obs,
    y_titles=...,
    x_titles=...,
    *,
    x_obs=None,
    y_obs=None,
    plotting_func=None,
    axes_func=None,
    tile_size=None,
    plot_kwds=None,
    label_side=None,
    title_side=None,
    fig=None,
    **grid_kwds,
):
    # 2d because obs is laid out over the grid: (rows, cols), either one broadcast
    obs = np.atleast_2d(Function.get_array(obs))
    # TODO support array of...
    x_obs = Function.get(x_obs)
    y_obs = Function.get(y_obs)

    # cbar positioning defaults: one bar per distinct obs
    rows, cols = obs.shape[:2]
    cbar_mode, cbar_location = "single", "right"
    if rows > 1 and cols > 1:
        cbar_mode = "each"
    elif rows > 1:
        cbar_mode = "edge"
    elif cols > 1:
        cbar_mode = "edge"
        cbar_location = "bottom"
    _prepare_grid_kwds(
        grid_kwds, aspect=True, cbar_mode=cbar_mode, cbar_location=cbar_location
    )
    cbar_mode, cbar_location = grid_kwds["cbar_mode"], grid_kwds["cbar_location"]
    # a bar of its own already labels the obs; shared ones leave it to the titles
    label_cbar = obs.size == 1 or cbar_mode == "each"
    edge_x = cbar_mode == "edge" and cbar_location in {"top", "bottom"}
    edge_y = cbar_mode == "edge" and cbar_location in {"left", "right"}

    # OPT y_obs, x_obs from xg? dunno what obs does with dims >= 2
    dims = dict(zip(REPORT_2D_DIMS, [y_titles, x_titles]))
    xg, br_arrays, titles = _report_dims(ug, dims, [obs])
    br_ug = br_arrays[0]  # OPT do we really need the broadcasted ug here?
    fig, grid, sides = _report_grid(
        xg,
        br_ug.shape[:2],
        titles,
        [obs],
        tile_size,
        fig,
        grid_kwds,
        label_side,
        title_side,
    )

    # normalization
    # TODO handle norm given in im_kwds
    if cbar_mode in {"single", "edge"}:
        agg_axs = (0, 1)
        if cbar_mode == "edge":
            agg_axs = 0 if cbar_location in {"top", "bottom"} else 1
        obs = autoscale_norms(obs, br_ug, agg_axs)

    axes_func = axes_func or axes_2d
    plot_kwds = deepcopy(plot_kwds) or {}
    it = np.nditer(
        [grid.axes_row, obs], ["refs_ok", "multi_index"], op_axes=[[0, 1]] * 2
    )
    for ax, o in it:
        ax, o = ax.item(), o.item()
        axes_func(br_ug[it.multi_index], o, x_obs, y_obs, plotting_func, plot_kwds, ax)
        if (cbar := getattr(getattr(ax, "cax", None), "cbar", None)) and not label_cbar:
            cbar.set_label("")
    grid.obs = obs
    _report_titles(
        fig,
        grid,
        br_ug,
        titles,
        sides,
        tile_size,
        obs,
        (label_cbar or edge_x, label_cbar or edge_y),
    )
    return fig, grid


def uniq_flat_mask(dat):
    dat = np.ma.ravel(dat)  # w/o ma drops mask
    return np.unique(dat[~dat.mask])


def sm_from_obs(obs, us=None):
    norm = getattr(obs, "norm", None)  # TODO from scale? see mpl.Colorizer.norm (!)
    cmap = getattr(obs, "cmap", None)
    if us is not None:
        # copy because ScalarMappable(norm=my_norm).norm is my_norm
        norm = copy(norm) if norm else colors.Normalize()
        cmap = copy(plt.get_cmap(cmap))
        uniq = uniq_flat_mask(obs(us))
        if getattr(obs, "digitize", uniq.size < MAX_DIGITIZED):
            cmap = cmap.resampled(uniq.size)
            cbar_kwds = obs.setdefault("cbar_kwds", {})
            try:
                # TODO generalize to log and unevenly spaced values
                grid = LinearGrid.from_points(uniq)
            except ValueError:
                # alternatively: NoNorm, but then sm.to_rgba(z) gives wrong color
                norm = colors.BoundaryNorm(bin_edges(uniq), uniq.size)
                cbar_kwds.setdefault("ticks", ticker.FixedLocator(uniq))
            else:
                # norm.autoscale_None(grid.extent) # FIXME do not resample then
                norm.vmin, norm.vmax = grid.extent
                cbar_kwds.setdefault(
                    "ticks", ticker.MultipleLocator(grid.step, uniq[0])
                )
        else:
            norm.autoscale_None(uniq)
    return cm.ScalarMappable(norm, cmap)


def _cbar_obs_grid(cbar_obs, ug, grid_kwds):
    """A copy of `cbar_obs` per tile, its `sm` scaled over the tiles sharing its
    colorbar: the whole grid, the row or column an edge bar serves, or the tile."""
    agg = {"single": (0, 1), "each": ()}.get(grid_kwds["cbar_mode"])
    if agg is None:  # edge: one bar per row, or per column
        agg = (1,) if grid_kwds["cbar_location"] in {"left", "right"} else (0,)
    obs_grid, done = np.empty(ug.shape[:2], object), {}
    for ij in np.ndindex(*obs_grid.shape):
        key = tuple(slice(None) if k in agg else i for k, i in enumerate(ij))
        if key not in done:
            done[key] = obs = copy(cbar_obs)
            obs.sm = sm_from_obs(cbar_obs, ug[key])
        obs_grid[ij] = done[key]
    return obs_grid


def add_cbar(ax, mappable, obs):
    """Colorbar for `ax` on the cax the grid gave it, once per cax: neighbouring tiles
    may well share one. Multiple cbars give problems when extend != "neither"."""
    cax = getattr(ax, "cax", None)
    if cax is None or not cax.get_visible() or hasattr(cax, "cbar"):
        return None
    # setp() does not work with Colorbar
    cbar_kwds = {"label": obs} | _parse_obs_props(obs, "cbar_kwds")
    cax.cbar = cax.colorbar(mappable, **cbar_kwds)
    return cax.cbar


def autoscale_norms(obs, ug, agg_axs=None):
    obs = np.squeeze(obs, tuple(range(2, np.ndim(obs))))
    if all(getattr(o, "norm", None) and o.norm.scaled() for o in obs.ravel()):
        return obs
    br = np.broadcast_shapes(*[a.shape[:2] for a in (obs, ug)])
    ug = np.broadcast_to(ug, br + ug.shape[2:])
    vmins = np.ma.masked_all(br)
    vmaxs = np.ma.masked_all(br)
    for ij, o in np.ndenumerate(np.broadcast_to(obs, br)):
        us = ug[ij]
        if o and np.any(us != ""):  # takes masks into account
            try:
                dat = o(us)  # OPT inefficient?
            except ValueError:
                pass
            else:
                # these function seem to respect masks as well
                vmins[ij] = np.nanmin(dat)
                vmaxs[ij] = np.nanmax(dat)
    vmins = np.nanmin(vmins, axis=agg_axs, keepdims=True)
    vmaxs = np.nanmax(vmaxs, axis=agg_axs, keepdims=True)
    obs_scaled = []
    br = np.broadcast(obs, vmins, vmaxs)
    # copy after broadcasting to handle multiple norms for same obs
    for o, vmin, vmax in br:
        if o:
            o = copy(o)
            try:
                norm = copy(o.norm)
            except AttributeError:
                norm = None
            o.norm = norm or colors.Normalize()
            o.norm.autoscale_None([vmin, vmax])
        obs_scaled.append(o)
    return np.reshape(obs_scaled, br.shape)


def join_obs_names(*obs, sep=",", junc="_vs_"):
    return junc.join(sep.join(o.name for o in np.ravel(os) if o) for os in obs)


def sanitize_fig_name(fig):
    fig.set_label(sanitize_path(fig.get_label()))


def _titled(title):
    """Whether `grid_titles` will draw anything for `title`. `...` reaches it only
    from the non-xarray path of `_prepare_uids_grid`, which has no coords to name
    the rows/columns with -- elsewhere `transpose_grid` has already resolved it."""
    return title not in {None, ...}


def _title_props():
    """The text properties a title is drawn with, as `Axes.set_title` reads them."""
    props = {
        "fontsize": rcParams["axes.titlesize"],
        "fontweight": rcParams["axes.titleweight"],
    }
    if str(rcParams["axes.titlecolor"]).lower() != "auto":
        props["color"] = rcParams["axes.titlecolor"]
    return props


def grid_titles(axs, pos, ug=None, title=None, obs=None, has_cbar=None, past_cbar=True):
    vertical = pos in {"left", "right"}
    ug = np.atleast_2d(ug)
    obs = np.atleast_2d(obs)
    if not vertical:
        ug = ug.swapaxes(0, 1)
        obs = obs.swapaxes(0, 1)
    obs_titles = []
    try:
        obs = np.squeeze(obs, tuple(range(2, np.ndim(obs))))
    except ValueError:
        pass
    else:
        if obs.shape[0] > 1 and obs.shape[1] == 1:
            obs_titles = [o.string() for o in obs[:, 0]]
    ug_titles = []
    if _titled(title):
        title = np.atleast_1d(Function.get_array(title))
        ug = np.ma.masked_equal(ug, "")
        # TODO mixed Function/Measure?
        if isinstance(title.item(0), Measure):
            ug = [us.flat for us in ug]
            title_func = Measure.strings
        else:  # assume function, buffer axis uids
            title_func = Function.strings
        ug = [next(filterfalse(np.ma.is_masked, us_ax), "") for us_ax in ug]
        ug_titles = [title_func(title, u, junc="\n") if u else "" for u in ug]
    if not has_cbar and obs_titles:
        if ug_titles:
            ug_titles = map("\n".join, np.broadcast(obs_titles, ug_titles))
        else:
            ug_titles = obs_titles
    # one pad for the whole row/column, so that the titles line up
    pad = rcParams["axes.titlepad"] + _decorations_pad(axs, pos, past_cbar)
    for ax, t in zip(axs, ug_titles):
        label = add_axis_label(ax, t, pos, labelpad=pad, **_title_props())
        setattr(ax, ("y" if vertical else "x") + "_title", label)


def add_axis_label(ax, label, loc, fontdict=None, labelpad=None, **kwargs):
    """Add a second axis-label-like text using ax.text with an absolute pad
    in points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to label.
    label : str
        Text of the label.
    fontdict : dict, optional
        A dictionary to override the default text properties.
    labelpad : float, optional
        Padding in points away from the axes. If None, uses rcParams['axes.labelpad'].
    loc : {'left', 'right', 'top', 'bottom'}, default: 'right'
        Which side of the axes to place the label.
        'top'/'bottom' behave like an x-label; 'left'/'right' like a y-label.
    **kwargs
        Additional keyword arguments forwarded to `ax.text`.

    Returns
    -------
    text : matplotlib.text.Text
        The created text object.

    """
    fig = ax.figure

    if labelpad is None:
        labelpad = rcParams["axes.labelpad"]  # in points

    labelpad = labelpad / 72.0  # ScaledTranslation expects inches
    # the pad is from the axes frame: to clear ticks, tick labels and axis label,
    # pass labelpad from _decorations_pad (as grid_titles does)

    if loc == "bottom":
        x, y = 0.5, 0.0
        dx, dy = 0.0, -labelpad
        defaults = {"ha": "center", "va": "top"}
    elif loc == "top":
        x, y = 0.5, 1.0
        dx, dy = 0.0, labelpad
        defaults = {"ha": "center", "va": "bottom"}
    elif loc == "left":
        x, y = 0.0, 0.5
        dx, dy = -labelpad, 0.0
        defaults = {"ha": "right", "va": "center", "rotation": 90}
    elif loc == "right":
        x, y = 1.0, 0.5
        dx, dy = labelpad, 0.0
        defaults = {"ha": "left", "va": "center", "rotation": 90}
    else:
        raise ValueError("loc must be one of 'left', 'right', 'top', 'bottom'")

    # Collect text properties: defaults < fontdict < kwargs
    text_kwargs = {}
    text_kwargs.update(defaults)
    if fontdict is not None:
        text_kwargs.update(fontdict)
    text_kwargs.update(kwargs)

    if "fontsize" not in text_kwargs and "size" not in text_kwargs:
        text_kwargs["fontsize"] = rcParams["axes.labelsize"]

    trans = ax.transAxes + transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    text = ax.text(x, y, label, transform=trans, **text_kwargs)
    return text


def annotate_image_axis(axis: mpl.axis.Axis, obs: None | Function, u: str):
    if not obs:
        return
    axis.set_label_text(obs)
    vals = np.ma.asanyarray(obs(u))  # ensure has mask attribute
    if axis.axis_name == "x":
        vals = vals.swapaxes(0, 1)
    try:
        vals = [np.unique(vs[~vs.mask]).item() for vs in vals]
    except ValueError:
        logger.warning(f"Could not set image {axis.axis_name}-ticks")
    else:
        axis.set_ticks(np.arange(len(vals)))
        axis.set_ticklabels([f"${v}$" for v in vals])
