import logging
from copy import copy, deepcopy
from itertools import filterfalse

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axg
import numpy as np
from matplotlib import cm, colors, rcParams, ticker, transforms

from simsio.analysis.grids import LinearGrid, UniformGrid, bin_edges
from simsio.analysis.organize import nest_grids, transpose_grid
from simsio.analysis.quantities import Function, Measure
from simsio.analysis.utils import sanitize_path

__all__ = [
    "annotate_image_axis",
    "apply_obs_props",
    "autoscale_norms",
    "axes_1d",
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
    y_side=None,
    fig=None,
    **grid_kwds,
):
    defaults = _prepare_grid_kwds(grid_kwds)
    # axg.Grid does not supprt cbar-related args
    # we nonetheless keep them in grid_kwds in analogy with report_2d
    # TODO same for other args?
    cbar_size = grid_kwds.pop("cbar_size", CBAR_SIZE)
    cbar_pad = grid_kwds.pop("cbar_pad", AXES_PAD)

    # DEL cycler=None hack while cycler not implemented
    dims = dict(zip(REPORT_1D_DIMS, [y_titles, x_titles, None, cbar_obs]))
    xg, ug, (row_titles, col_titles, _, cbar_obs) = _prepare_uids_grid(ug, dims)
    br_arrays = _prepare_arrays(len(dims), [ug, x_obs, y_obs])  # TODO x/y order, cbar
    ug = br_arrays[0]
    # y labels opposite the row titles, which grid_titles puts on the left
    y_side = y_side or ("right" if _titled(row_titles) else "left")

    # figure & axes
    shape = ug.shape[:2]
    size = _fig_size(shape, tile_size, grid_kwds)
    label = _fig_name(xg, [y_obs, x_obs, cbar_obs])
    fig = plt.figure(fig or label, size)
    # mpl's label_mode is hardwired to the bottom-left: use ours unless asked
    if own_labels := "label_mode" not in grid_kwds:
        grid_kwds["label_mode"] = "keep"
    grid = axg.Grid(fig, 111, shape, **grid_kwds)
    if own_labels:
        _label_grid(grid, grid_kwds, y_side)

    axes_func = axes_func or axes_1d
    plot_kwds = deepcopy(plot_kwds) or {}
    if cbar_obs:  # TODO support multiple colorbars
        div = grid.get_divider()
        grid.cbar_pad = axg.Size.Fixed(cbar_pad)  # grown by _fit_axes_pad
        div.append_size("right", grid.cbar_pad)
        div.append_size("right", axg.Size.Fixed(cbar_size))
        cax_locator = div.new_locator(nx=-2, nx1=-1, ny=0, ny1=-1)
        grid.cax = fig.add_subplot(axes_locator=cax_locator)
        cbar_obs = copy(cbar_obs)
        cbar_obs.sm = sm_from_obs(cbar_obs, ug)

    grid_iter, plot_iter = np.nested_iters(
        br_arrays, [[0, 1], [2, 3]], ["refs_ok", "multi_index"], order="C"
    )
    for ax, _ in zip(grid, grid_iter):
        us = ug[grid_iter.multi_index]
        axes_func(us, plot_iter, cbar_obs, plotting_func, plot_kwds, ax)
        if ug.shape[2] > 1:
            ax.legend()  # TODO cycler
    if cbar_obs:
        cbar_kwds = getattr(cbar_obs, "cbar_kwds", {})
        fig.colorbar(cbar_obs.sm, cax=grid.cax, label=cbar_obs, **cbar_kwds)
    for ax in grid:
        if not ax.lines:
            ax.axis("off")
    # an explicit axes_pad is left alone
    if "axes_pad" in defaults and _needs_pad_fit(grid, grid_kwds):
        _fit_axes_pad(fig, grid, grid_kwds, tile_size)
    # after the fit: the titles clear the decorations of the final layout
    grid_titles(grid.axes_row[0], "top", ug, col_titles)
    grid_titles(grid.axes_column[0], "left", ug, row_titles)
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
    try:
        cax = ax.cax
    except AttributeError:
        pass
    else:
        # multiple cbars give problems when extend != "neither"
        if not hasattr(cax, "cbar"):
            # setp() does not work with Colorbar
            cbar_kwds = _parse_obs_props(obs, "cbar_kwds")
            cbar_kwds.setdefault("label", obs)
            cax.cbar = ax.cax.colorbar(im, **cbar_kwds)
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


def _shared_axes(grid_kwds):
    """(share_x, share_y) as axg.Grid will understand them."""
    share_all = grid_kwds.get("share_all", False)
    return (
        share_all or grid_kwds.get("share_x", True),
        share_all or grid_kwds.get("share_y", True),
    )


def _prepare_grid_kwds(grid_kwds, cbar=True):
    """Fill in the grid defaults, returning the keys they were needed for."""
    given = set(grid_kwds)
    # unshared axes need more, but that is measured afterwards (_fit_axes_pad)
    grid_kwds.setdefault("axes_pad", AXES_PAD)
    if cbar:
        grid_kwds.setdefault("cbar_pad", AXES_PAD)
        grid_kwds.setdefault("cbar_size", CBAR_SIZE)
    return set(grid_kwds) - given


SIDES = ("left", "bottom", "right", "top")


def _label_grid(grid, grid_kwds, y_side="left", x_side="bottom"):
    """Ticks on the given sides of the grid, axis labels on the edge tile there,
    tick labels on that edge too -- or on every tile of an unshared direction,
    each having its own scale. Replaces Grid's label_mode, which can only label
    the bottom row and the left column."""
    nrows, ncols = grid.get_geometry()
    edges = {"top": 0, "bottom": nrows - 1, "left": 0, "right": ncols - 1}
    shared = dict(zip((x_side, y_side), _shared_axes(grid_kwds)))
    for (i, j), ax in np.ndenumerate(np.array(grid.axes_row, object)):
        for axis, side, k in ((ax.xaxis, x_side, i), (ax.yaxis, y_side, j)):
            axis.set_ticks_position(side)  # also moves the tick labels
            axis.set_label_position(side)
            edge = k == edges[side]
            axis.set_tick_params(**{"label" + side: edge or not shared[side]})
            axis.label.set_visible(edge)  # set later by axes_1d, but not shown


def _overhangs(ax):
    """Room (inches) the decorations of `ax` take outside its frame, ordered as
    SIDES. Cheaper than a draw: get_tightbbox() runs the locators and places the
    labels by itself."""
    bb = ax.get_tightbbox()
    fr = ax.bbox
    if bb is None:  # invisible axes
        return np.zeros(4)
    over = [fr.x0 - bb.x0, fr.y0 - bb.y0, bb.x1 - fr.x1, bb.y1 - fr.y1]
    return np.clip(over, 0, None) / ax.figure.dpi


def _decorations_pad(axs, pos):
    """Points to clear whatever `axs` already draw on their `pos` side. Shared by
    the whole row/column, so that what is placed past it stays aligned."""
    over = max(_overhangs(ax)[SIDES.index(pos)] for ax in axs)
    return 72 * over  # ScaledTranslation and labelpad work in points


def _needs_pad_fit(grid, grid_kwds):
    """Whether anything can reach into the gaps between the tiles: tick labels,
    which unshared axes carry on every tile (_label_grid), or an offset text,
    which any tile can carry above (y) or right of (x) its frame. The colorbar
    pad is a gap too, and the last column may well label its right side."""
    if hasattr(grid, "cbar_pad") and grid[0].yaxis.get_label_position() == "right":
        return True
    if grid.get_geometry() == (1, 1):  # no gaps
        return False
    if not all(_shared_axes(grid_kwds)):
        return True
    return any(_has_offset(axis) for ax in grid for axis in (ax.xaxis, ax.yaxis))


def _has_offset(axis):
    fmt = axis.get_major_formatter()
    fmt.set_locs(axis.get_majorticklocs())  # as a draw would, but for free
    return bool(fmt.get_offset())


def _fit_axes_pad(fig, grid, grid_kwds, tile_size):
    """Set axes_pad to AXES_PAD plus the room the tiles need between them, and
    rescale the figure so that they keep their nominal size. Single pass: the pad
    is *not* fitted again if the caller later reformats the axes."""
    over = np.array([[_overhangs(ax) for ax in row] for row in grid.axes_row])
    # each gap must fit what the two tiles it separates reach into it
    h_pad = AXES_PAD + (over[:, :-1, 2] + over[:, 1:, 0]).max(initial=0)
    v_pad = AXES_PAD + (over[:-1, :, 1] + over[1:, :, 3]).max(initial=0)
    grid_kwds["axes_pad"] = (h_pad, v_pad)
    grid.set_axes_pad(grid_kwds["axes_pad"])
    cbar_pad = getattr(grid, "cbar_pad", None)
    # the colorbar too must clear what the last column draws on its right
    grown = over[:, -1, 2].max() if cbar_pad else 0.0
    if grown:
        cbar_pad.fixed_size += grown
    # the figure never accounted for the colorbar: at least do not shrink the tiles
    fig.set_size_inches(
        _fig_size(grid.get_geometry(), tile_size, grid_kwds) + [grown, 0]
    )


def _fig_size(shape, tile_size, grid_kwds):
    # just an estimate
    pad = np.asarray(grid_kwds["axes_pad"], float)  # (horizontal, vertical)
    return np.flip(shape[:2]) * (tile_size or TILE_SIZE + pad)


def _fig_name(xg, label_obs):
    return getattr(xg, "name", "") + "/" + join_obs_names(*label_obs)


def report_2d(
    ug,
    obs,
    *,
    y_titles=None,
    x_titles=None,
    x_obs=None,
    y_obs=None,
    plotting_func=None,
    tile_size=None,
    im_kwds=None,
    fig=None,
    **grid_kwds,
):
    im_kwds = im_kwds or {}
    _prepare_grid_kwds(grid_kwds)

    obs = Function.get_array(obs)
    # TODO support array of...
    x_obs = Function.get(x_obs)
    y_obs = Function.get(y_obs)

    plotting_func = plotting_func or plot_2d_data

    # cbar positioning defaults
    cbar_mode = "single"
    cbar_location = "right"
    if obs.ndim == 2:
        cbar_mode = "each"
    elif obs.shape[0] > 1:
        cbar_mode = "edge"
    elif obs.shape[1] > 1:
        cbar_mode = "edge"
        cbar_location = "bottom"
    cbar_mode = grid_kwds.setdefault("cbar_mode", cbar_mode)
    cbar_location = grid_kwds.setdefault("cbar_location", cbar_location)

    # OPT y_obs, x_obs from xg? dunno what obs does with dims >= 2
    dims = dict(zip(REPORT_2D_DIMS, [y_titles, x_titles]))
    xg, ug, (row_titles, col_titles) = _prepare_uids_grid(ug, dims)
    br_arrays = _prepare_arrays(len(dims), [ug, obs])

    # figure & axes
    shape = br_arrays[0].shape[:2]
    size = _fig_size(shape, tile_size, grid_kwds)
    label = _fig_name(xg, [obs])
    fig = plt.figure(fig or label, size)
    grid = axg.ImageGrid(fig, 111, shape, **grid_kwds)

    br_ug, _ = br_arrays  # OPT do we really need the broadcasted ug here?
    label_cbar = obs.size == 1 or (obs.ndim == 2 and cbar_mode == "each")
    # y-titles
    axs = grid.axes_column[-1 if cbar_location == "left" else 0]
    pos = "right" if cbar_location == "left" else "left"
    label_cbar_row = cbar_mode == "edge" and cbar_location in {"left", "right"}
    grid_titles(axs, pos, br_ug, row_titles, obs, label_cbar or label_cbar_row)
    # x-titles
    axs = grid.axes_row[-1 if cbar_location == "top" else 0]
    pos = "bottom" if cbar_location == "top" else "top"
    label_cbar_col = cbar_mode == "edge" and cbar_location in {"top", "bottom"}
    grid_titles(axs, pos, br_ug, col_titles, obs, label_cbar or label_cbar_col)

    # normalization
    # TODO handle norm given in im_kwds
    if cbar_mode in {"single", "edge"}:
        if cbar_mode == "single":
            lims_agg_axs = (0, 1)
        if cbar_mode == "edge":
            lims_agg_axs = 0 if cbar_location in {"top", "bottom"} else 1
        obs = autoscale_norms(obs, ug, lims_agg_axs)

    label_cbar = label_cbar or label_cbar_row or label_cbar_col
    it = np.nditer(
        [grid.axes_row, obs], ["refs_ok", "multi_index"], op_axes=[[0, 1]] * 2
    )
    for ax, o in it:
        ax = ax.item()
        o = o.item()
        us = br_ug[it.multi_index]
        plotting_func(o, us, x_obs=x_obs, y_obs=y_obs, ax=ax, **im_kwds)
        if cbar := getattr(ax, "cbar", None) and not label_cbar:
            cbar.set_label("")
    grid.obs = obs
    fig.align_labels()
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


def grid_titles(axs, pos, ug=None, title=None, obs=None, has_cbar=None):
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
    pad = rcParams["axes.labelpad"] + _decorations_pad(axs, pos)
    for ax, t in zip(axs, ug_titles):
        label = add_axis_label(ax, t, pos, labelpad=pad)
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
        text_kwargs["fontsize"] = rcParams["axes.labelsize"]  # TODO titlesize

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
