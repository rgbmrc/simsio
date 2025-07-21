import mplotter as plotter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import colors, ticker

from ..analysis import Function, Measure

TILE_SIZE = 1.33
AXES_PAD = 0.1
CBAR_SIZE = 0.1
MAX_DIGITIZED = 12

gr = ""


def fix_mfc(l):
    c = l.get_color()
    l.set(mfc=colors.to_rgba(c, 0.2), mec=colors.to_rgba(c, 1.0))
    return l


def sanitize_fig_name(fig):
    # TODO platform dependent, improve
    fig.set_label(fig.get_label().replace(":", "!").replace("/", "_")[:128])  # FIXME


def save_fig(fig=None, dest=None):
    fig = fig or plt.gcf()
    if dest:
        plotter.save_fig(fig, dest)
    else:
        sanitize_fig_name(fig)
        with fig_dir(gr):
            plotter.save_fig(fig)


def sm_from_obs(obs, us=None):
    # generalize to log and unevenly spaced values
    # copy because ScalarMappable(norm=my_norm).norm is my_norm
    norm = getattr(obs, "norm", None)
    cmap = getattr(obs, "cmap", None)
    if us is not None:
        norm = copy(norm) if norm else colors.Normalize()
        cmap = copy(plt.get_cmap(cmap))
        dat = np.ma.ravel(obs(us))  # w/o ma drops mask
        uniq = np.unique(dat[~dat.mask])
        if getattr(obs, "digitize", uniq.size < MAX_DIGITIZED):
            cmap = cmap.resampled(uniq.size)
            cbar_kwds = obs.setdefault("cbar_kwds", {})
            try:
                grid = Grid1D.from_points(uniq)
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
            norm.autoscale_None(dat)
    return ScalarMappable(norm, cmap)


# on top of matplotlib ones
PROP_ALIASES = {"major_locator": "loc", "major_formatter": "fmt"}


def parse_aliases(kwds):
    for name, alias in PROP_ALIASES.items():
        if alias in kwds:
            if name in kwds:
                raise ValueError(f"Specified both '{name}' and '{alias}'")
            kwds[name] = kwds.pop(alias)
    return kwds


def parse_obs_props(obs, props_name, **props):
    props = getattr(obs, props_name, {}) | props
    # props = parse_aliases(props) # NOTE enable if needed
    return props


def apply_obs_props(obj, obs, props_name, **props):
    props = parse_obs_props(obs, props_name, **props)
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
    plot_kwds = parse_obs_props(y_obs, "plot_kwds", **plot_kwds)
    if x_obs:
        l = ax.plot(x, y, **plot_kwds)
    else:
        l = ax.plot(y, **plot_kwds)
    return l


def join_obs_names(*obs, sep=",", junc="_vs_"):
    return junc.join((sep.join((o.name for o in np.ravel(os) if o)) for os in obs))


def report_1d(
    ug,
    x_obs,
    y_obs,
    cbar_obs=None,
    x_titles=None,
    y_titles=None,
    *,
    plotting_func=None,
    axes_func=None,
    tile_size=None,
    plot_kwds=None,
    **grid_kwds,
):
    AX_LOOP_DIM = 2

    axes_func = axes_func or axes_1d
    tile_size = tile_size or 2 * TILE_SIZE
    grid_kwds.setdefault("axes_pad", AXES_PAD)
    ndim = max(4, np.ndim(ug))
    iter_arrays = (ug, x_obs, y_obs)
    iter_arrays = (ug, x_obs, y_obs) = [append_til_ndim(a, ndim) for a in iter_arrays]
    plot_kwds = deepcopy(plot_kwds) or {}
    label = grid_kwds.pop("label", None) or join_obs_names(y_obs, x_obs, cbar_obs)
    shape = np.broadcast_shapes(*(a.shape[:2] for a in iter_arrays))
    ug, *_ = np.broadcast_arrays(*iter_arrays)
    size = np.flip(shape) * tile_size + (AXES_PAD + CBAR_SIZE, 0)
    fig = plt.figure(label, size)
    grid = axg.Grid(fig, 111, shape, **grid_kwds)

    if cbar_obs:
        div = grid.get_divider()
        div.append_size("right", axg.Size.Fixed(AXES_PAD))
        div.append_size("right", axg.Size.Fixed(CBAR_SIZE))
        cax_locator = div.new_locator(nx=-2, nx1=-1, ny=0, ny1=-1)
        grid.cax = fig.add_subplot(axes_locator=cax_locator)
        cbar_obs = copy(cbar_obs)
        cbar_obs.sm = sm_from_obs(cbar_obs, ug)

    grid_iter, plot_iter = np.nested_iters(
        iter_arrays,
        [[0, 1], [2, 3]],
        ["refs_ok", "multi_index"],
        order="C",
    )
    for ax, _ in zip(grid, grid_iter):
        axes_func(
            ug[grid_iter.multi_index],
            plot_iter,
            cbar_obs,
            plotting_func,
            plot_kwds,
            ax,
        )
        if y_obs.shape[AX_LOOP_DIM] > 1:
            ax.legend()
    if cbar_obs:
        cbar_kwds = getattr(cbar_obs, "cbar_kwds", {})
        fig.colorbar(cbar_obs.sm, cax=grid.cax, label=cbar_obs, **cbar_kwds)
    for ax in grid:
        if not ax.lines:
            ax.axis("off")

    # FIXME this removes eventual x_obs & y_obs labels
    #       even when e.g. title should be "top" and label "bottom"
    grid_titles(grid.axes_row[0], "top", ug, x_titles)
    grid_titles(grid.axes_column[0], "left", ug, y_titles)

    return fig, grid


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


def plot_2d_data(obs, u=None, x_obs=None, y_obs=None, ax=None, **im_kwds):
    ax = ax or plt.gca()
    if not obs:
        ax.axis("off")
        return None
    im_kwds = getattr(obs, "im_kwds", {}) | im_kwds
    # prevent autoscale of the original observable's norm
    # copies still needed also in grid e.g. when cbar_mode="each"
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
            cbar_kwds = parse_obs_props(obs, "cbar_kwds")
            cbar_kwds.setdefault("label", obs)
            cax.cbar = ax.cax.colorbar(im, **cbar_kwds)
    annote_image_axis(ax.xaxis, x_obs, u)
    annote_image_axis(ax.yaxis, y_obs, u)
    return im


def annote_image_axis(axis: mpl.axis.Axis, obs: None | Function, u: str):
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


def report_2d(
    ug,
    obs,
    y_titles=None,
    x_titles=None,
    x_obs=None,
    y_obs=None,
    plotting_func=None,
    tile_size=None,
    label=None,
    im_kwds=None,
    fig=None,
    **grid_kwds,
):
    im_kwds = im_kwds or {}
    obs_ndim = np.ndim(obs)
    iter_arrays = (ug, obs)
    ndim = max(2, *(np.ndim(a) for a in iter_arrays))
    iter_arrays = (ug, obs) = [append_til_ndim(a, ndim) for a in iter_arrays]
    iter_arrays = _ug, _obs = np.broadcast_arrays(*iter_arrays)

    # obs = Measure.get(obs)
    plotting_func = plotting_func or plot_2d_data

    # cbar positioning defaults
    cbar_mode = "single"
    cbar_location = "right"
    if obs_ndim == 2:
        cbar_mode = "each"
    elif obs.shape[0] > 1:
        cbar_mode = "edge"
    elif obs.shape[1] > 1:
        cbar_mode = "edge"
        cbar_location = "bottom"
    cbar_mode = grid_kwds.setdefault("cbar_mode", cbar_mode)
    cbar_location = grid_kwds.setdefault("cbar_location", cbar_location)

    shape = _obs.shape[:2]
    tile_size = tile_size or TILE_SIZE
    grid_kwds.setdefault("axes_pad", AXES_PAD)
    grid_kwds.setdefault("cbar_size", CBAR_SIZE)
    label = label or ",".join((o.name for o in np.ravel(obs) if o))
    fig = fig or plt.figure(label, np.asanyarray(shape[::-1]) * tile_size)
    grid = axg.ImageGrid(fig, 111, shape, **grid_kwds)

    label_cbar = obs.size == 1 or (obs_ndim == 2 and cbar_mode == "each")
    # y-titles
    axs = grid.axes_column[-1 if cbar_location == "left" else 0]
    pos = "right" if cbar_location == "left" else "left"
    label_cbar_row = cbar_mode == "edge" and cbar_location in {"left", "right"}
    grid_titles(axs, pos, _ug, y_titles, obs, label_cbar or label_cbar_row)
    # x-titles
    axs = grid.axes_row[-1 if cbar_location == "top" else 0]
    pos = "bottom" if cbar_location == "top" else "top"
    label_cbar_col = cbar_mode == "edge" and cbar_location in {"top", "bottom"}
    grid_titles(axs, pos, _ug, x_titles, obs, label_cbar or label_cbar_col)

    # normalization
    # TODO handle norm given in im_kwds
    if cbar_mode in {"single", "edge"}:
        if cbar_mode == "single":
            lims_agg_axs = (0, 1)
        if cbar_mode == "edge":
            lims_agg_axs = 0 if cbar_location in {"top", "bottom"} else 1
        obs = autoscale_norms(obs, ug, lims_agg_axs)

    label_cbar = label_cbar or label_cbar_row or label_cbar_col
    it_flags = ["refs_ok", "multi_index"]
    it = np.nditer([grid.axes_row, obs], it_flags, op_axes=[[0, 1]] * 2)
    for ax, o in it:
        ax = ax.item()
        o = o.item()
        us = _ug[it.multi_index]
        plotting_func(o, us, x_obs=x_obs, y_obs=y_obs, ax=ax, **im_kwds)
        if cbar := getattr(ax, "cbar", None) and not label_cbar:
            cbar.set_label("")
    grid.obs = obs
    fig.align_labels()
    return fig, grid


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


def grid_titles(axs, pos, ug=None, title=None, obs=None, has_cbar=None):
    ug = np.atleast_2d(ug)
    obs = np.atleast_2d(obs)
    if pos in {"top", "bottom"}:
        ug = ug.swapaxes(0, 1)
        obs = obs.swapaxes(0, 1)
        xy = "x"
    else:
        xy = "y"
    obs_titles = []
    try:
        obs = np.squeeze(obs, tuple(range(2, np.ndim(obs))))
    except ValueError:
        pass
    else:
        if obs.shape[0] > 1 and obs.shape[1] == 1:
            obs_titles = [o.string() for o in obs[:, 0]]
    ug_titles = []
    if title is not None:
        title = np.atleast_1d(title)
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
    pad = mpl.rcParams["axes.titlepad"]
    for ax, t in zip(axs, ug_titles):
        # ax.text(0.5, 0.5, t, transform=ax.transAxes, ha="center", va="center")
        axis = getattr(ax, f"{xy}axis")
        axis.labelpad = pad
        axis.set_label_text(t)
        axis.set_label_position(pos)
        axis.label.set_visible(True)


def get_data_grids(dat, u=None):
    shape = np.shape(dat)  # works also for scalar
    if u:
        try:
            return [
                Grid1D(lat_N, (0, 1)).broadcast_to(dat_N)
                for lat_N, dat_N in zip(lvals(u), shape)
            ]
        except ValueError:
            logger.warning("could not broadcast data grid")
    return [Grid1D(dat_N, (0, 1)) for dat_N in shape]


def get_data_extent(data_grids):
    return [lim for grid in data_grids for lim in grid.extent]


def axes_image(ax):
    ax.set(
        xticks=[0.5],
        yticks=[0.5],
        xticklabels=[],
        yticklabels=[],
        xlim=(0, 1),
        ylim=(0, 1),
    )
    return ax


def plot_2d_local(obs, u=None, x_obs=None, y_obs=None, ax=None, **im_kwds):
    axes_image(ax)
    try:
        dat = obs(u)
        first_valid = next(filter(None, np.ravel(u)))  # HACK
    except (ValueError, StopIteration, TypeError):  # TypeError is foor o == None
        grids = None  # should never be accessed
    else:
        grids = get_data_grids(dat, first_valid)
        im_kwds.setdefault("extent", get_data_extent(grids))
        im_kwds.setdefault("origin", "lower")
    im = plot_2d_data(obs, u, x_obs=x_obs, y_obs=y_obs, ax=ax, **im_kwds)
    if im is not None:
        im.grids = grids
    return im


def report_single_2d(ug, obs, **report_2d_kwds):
    report_2d_kwds.setdefault("plotting_func", plot_2d_local)
    report_2d_kwds.setdefault("cbar_mode", "single")
    return report_2d(np.atleast_2d(ug), [obs], **report_2d_kwds)


def report_multi_2d(ug, obs, **report_2d_kwds):
    report_2d_kwds.setdefault("plotting_func", plot_2d_local)
    report_2d_kwds.setdefault("cbar_mode", "edge")
    return report_2d([ug], obs, **report_2d_kwds)
