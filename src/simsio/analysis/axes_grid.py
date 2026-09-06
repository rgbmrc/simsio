"""A grid of axes halfway between `axg.Grid` and `axg.ImageGrid`.

`axg.Grid` gives equally sized tiles and optional per-row/column sharing, but no
colorbars. `axg.ImageGrid` adds the colorbars, but hardwires the sharing on and sizes
its tiles by their data range. `AxesGrid` takes both halves, and owns the labelling so
that ticks and labels can sit on any side of the grid, not only the bottom-left one.

"""

import mpl_toolkits.axes_grid1 as axg
import numpy as np
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.axes_grid import _cbaraxes_class_factory

Size = axg.Size

__all__ = ["SIDES", "AxesGrid"]

SIDES = ("left", "bottom", "right", "top")
OPPOSITE = dict(zip(SIDES, SIDES[2:] + SIDES[:2]))


def parse_sides(sides, default=("bottom", "left")):
    """(x, y) sides from any subset of them, in any order: "right", ("top", "right")."""
    out = list(default)
    for side in (sides,) if isinstance(sides, str) else sides:
        if side not in SIDES:
            raise ValueError(f"{side!r} is not one of {SIDES}")
        out[side in {"left", "right"}] = side
    return tuple(out)


def overhangs(ax, *others):
    """Room (inches) the decorations of `ax`, and of anything in `others` drawn beside
    it, take outside its frame, ordered as SIDES. Cheaper than a draw: get_tightbbox()
    runs the locators and places the labels by itself."""
    bbs = [bb for bb in (a.get_tightbbox() for a in (ax, *others)) if bb is not None]
    if not bbs:  # invisible axes
        return np.zeros(4)
    bb, fr = Bbox.union(bbs), ax.bbox
    over = [fr.x0 - bb.x0, fr.y0 - bb.y0, bb.x1 - fr.x1, bb.y1 - fr.y1]
    return np.clip(over, 0, None) / ax.figure.dpi


def tile_overhangs(ax):
    """`overhangs(ax)` including the colorbar the grid gave `ax` alone, if it has one:
    the two are laid out as a unit, so whatever the bar draws sits outside the tile."""
    cax = getattr(ax, "cax", None)
    return overhangs(ax, cax) if getattr(cax, "tile", None) is ax else overhangs(ax)


def has_offset(axis):
    fmt = axis.get_major_formatter()
    fmt.set_locs(axis.get_majorticklocs())  # as a draw would, but for free
    return bool(fmt.get_offset())


class AxesGrid(axg.ImageGrid):
    """Grid of axes with optional colorbars, sharing and label sides of our own.

    Parameters are `axg.ImageGrid`'s, minus the deprecated `ngrids`, plus

    share_x, share_y : bool, default: True
        Share the x-axis down each column and the y-axis along each row, as
        `axg.Grid` does -- `axg.ImageGrid` forces both on.
    label_sides : side or (side, side), default: ("bottom", "left")
        Sides carrying the ticks, tick labels and axis labels, given in any order.
    label_mode : {"L", "1", "all", "keep"}, default: "L"
        Which tiles are labelled on those sides: the edge row and column ("L"), only
        the tile where they meet ("1"), all of them, or leave them alone ("keep").
        A direction that is not shared always labels every tile: each has its own
        scale.

    """

    def __init__(
        self,
        fig,
        rect,
        nrows_ncols,
        direction="row",
        axes_pad=0.02,
        *,
        share_all=False,
        share_x=True,
        share_y=True,
        aspect=False,
        label_mode="L",
        label_sides=("bottom", "left"),
        cbar_mode=None,
        cbar_location="right",
        cbar_pad=None,
        cbar_size="5%",
        cbar_set_cax=True,
        axes_class=None,
    ):
        if cbar_mode not in {"each", "single", "edge", None}:
            raise ValueError(f"unknown cbar_mode {cbar_mode!r}")
        if cbar_location not in SIDES:
            raise ValueError(f"cbar_location must be one of {SIDES}")
        self._colorbar_mode = cbar_mode  # read back by _init_locators
        self._colorbar_location = cbar_location
        self._colorbar_pad = cbar_pad
        self._colorbar_size = cbar_size
        self._share = (share_all or share_x, share_all or share_y)
        self._label_mode, self._label_sides = "keep", parse_sides(label_sides)
        # ImageGrid.__init__ hardwires share_x/share_y: only Grid's honours them
        axg.Grid.__init__(
            self,
            fig,
            rect,
            nrows_ncols,
            direction=direction,
            axes_pad=axes_pad,
            share_all=share_all,
            share_x=share_x,
            share_y=share_y,
            aspect=aspect,
            label_mode="keep",
            axes_class=axes_class,
        )
        for cax in self.cbar_axes:
            fig.add_axes(cax)
        if cbar_set_cax:
            for i, ax in enumerate(self.axes_all):
                ax.cax = self.cbar_axes[self._cbar_index(i)]
                if cbar_mode == "each":
                    ax.cax.tile = ax  # laid out as one unit, see tile_overhangs
        self.set_label_mode(label_mode, label_sides)

    def _cbar_index(self, i):
        """Which colorbar serves tile `i`: its own, its row's or column's, or the one."""
        col, row = self._get_col_row(i)
        if self._colorbar_mode == "single":
            return 0
        if self._colorbar_mode == "edge":
            return row if self._colorbar_location in {"left", "right"} else col
        return i

    def _tile_size(self, ax, axis):
        """Equal tiles, as axg.Grid; sized by the data limits when the aspect is
        locked, as axg.ImageGrid, so that images tile without gaps."""
        if not self.get_aspect():
            return Size.Scaled(1)
        return (Size.AxesX, Size.AxesY)[axis](
            ax, aspect="axes", ref_ax=self.axes_all[0]
        )

    def _sizes(self, axis):
        """Divider sizes along `axis` (0 horizontal, 1 vertical), with the positions of
        the tiles and of the colorbars in the list. Vertical runs bottom-up."""
        axs = self.axes_row[0] if axis == 0 else self.axes_column[0][::-1]
        pad = (self._horiz_pad_size, self._vert_pad_size)[axis]
        mode, loc = self._colorbar_mode, self._colorbar_location
        ours = loc in (("left", "right"), ("bottom", "top"))[axis]
        first = loc in {"left", "bottom"}  # the bar precedes the tile it serves
        sizes, tiles, cbars = [], [], []
        for i, ax in enumerate(axs):
            if i:
                sizes.append(pad)
            size = self._tile_size(ax, axis)
            edge = i == (0 if first else len(axs) - 1)
            bar = ours and (mode == "each" or (mode == "edge" and edge))
            if bar and first:
                cbars.append(len(sizes))
                sizes += [Size.from_any(self._colorbar_size, size), self._cbar_pad_size]
            tiles.append(len(sizes))
            sizes.append(size)
            if bar and not first:
                sizes.append(self._cbar_pad_size)
                cbars.append(len(sizes))
                sizes.append(Size.from_any(self._colorbar_size, size))
        if ours and mode == "single":  # one bar beside the whole grid
            # a relative cbar_size refers to the grid extent along `axis` (mpl uses the
            # other one), which is what a "5%"-wide bar next to the grid should mean
            ref = len(axs) * (Size.AxesX, Size.AxesY)[axis](self.axes_llc)
            slot = [self._cbar_pad_size, Size.from_any(self._colorbar_size, ref)]
            if first:  # size then pad, so that no axes_pad is added on top
                sizes[:0] = slot[::-1]
                tiles = [t + 2 for t in tiles]
                cbars = [0]
            else:
                cbars = [len(sizes) + 1]
                sizes += slot
        return sizes, tiles, cbars

    def _init_locators(self):
        # called by Grid.__init__, which is also where cbar_axes must come to life
        div = self._divider
        mode, loc = self._colorbar_mode, self._colorbar_location
        vertical = loc in {"bottom", "top"}
        if self._colorbar_pad is None:
            self._colorbar_pad = (self._horiz_pad_size, self._vert_pad_size)[
                vertical
            ].fixed_size
        # one object for every slot, so that fit_axes_pad grows them all at once
        self._cbar_pad_size = Size.Fixed(self._colorbar_pad)
        cax_class = _cbaraxes_class_factory(self._defaultAxesClass)
        fig = self.axes_all[0].get_figure(root=False)
        self.cbar_axes = [
            cax_class(fig, div.get_position(), orientation=loc) for _ in self.axes_all
        ]
        h, h_ax, h_cb = self._sizes(0)
        v, v_ax, v_cb = self._sizes(1)
        for i, ax in enumerate(self.axes_all):
            col, row = self._get_col_row(i)
            ny = self._nrows - 1 - row  # divider rows count from the bottom
            ax.set_axes_locator(div.new_locator(nx=h_ax[col], ny=v_ax[ny]))
            cax, visible = self.cbar_axes[self._cbar_index(i)], True
            if mode == "each":
                nx, ny_ = (h_ax[col], v_cb[ny]) if vertical else (h_cb[col], v_ax[ny])
                cax.set_axes_locator(div.new_locator(nx=nx, ny=ny_))
            elif mode == "edge":
                if vertical:
                    cax.set_axes_locator(div.new_locator(nx=h_ax[col], ny=v_cb[0]))
                else:
                    cax.set_axes_locator(div.new_locator(nx=h_cb[0], ny=v_ax[ny]))
                visible = i < (self._ncols if vertical else self._nrows)
            elif mode == "single":
                if vertical:
                    cax.set_axes_locator(div.new_locator(nx=0, nx1=-1, ny=v_cb[0]))
                else:
                    cax.set_axes_locator(div.new_locator(nx=h_cb[0], ny=0, ny1=-1))
                visible = i == 0
            else:
                visible = False
                self.cbar_axes[i].set_position([1, 1, 0.001, 0.001], which="active")
            self.cbar_axes[i].set_visible(visible)
        div.set_horizontal(h)
        div.set_vertical(v)

    def set_label_mode(self, mode=None, sides=None):
        """Move the ticks, tick labels and axis labels, and choose the tiles carrying
        them. Either argument can be updated alone; see the class docstring. Replaces
        Grid's, which is hardwired to the bottom row and the left column."""
        self._label_mode = mode = mode or self._label_mode
        if sides is not None:
            self._label_sides = parse_sides(sides)
        if mode == "keep":
            return
        if mode not in {"L", "1", "all"}:
            raise ValueError(f"unknown label_mode {mode!r}")
        nrows, ncols = self.get_geometry()
        edge = {"top": 0, "bottom": nrows - 1, "left": 0, "right": ncols - 1}
        for (i, j), ax in np.ndenumerate(np.array(self.axes_row, object)):
            at_edge = [k == edge[side] for k, side in zip((i, j), self._label_sides)]
            for n, axis in enumerate((ax.xaxis, ax.yaxis)):
                side = self._label_sides[n]
                axis.set_ticks_position(side)  # also moves the tick labels
                axis.set_label_position(side)
                # an unshared direction labels every tile: each has its own scale
                on = (
                    at_edge[n] if mode == "L" else all(at_edge) if mode == "1" else True
                )
                axis.set_tick_params(**{"label" + side: on or not self._share[n]})
                axis.label.set_visible(at_edge[n] if mode == "L" else on)

    def needs_pad_fit(self):
        """Whether anything can reach into the gaps between the tiles: tick labels,
        which unshared axes carry on every tile, or an offset text, which any tile can
        carry above (y) or right of (x) its frame. A colorbar pad is a gap too, and the
        tiles may well label the side it sits on."""
        if self._colorbar_mode == "each":  # a bar in every gap, all of them labelled
            return True
        if self._colorbar_mode and self._colorbar_location in self._label_sides:
            return True
        if self.get_geometry() == (1, 1):  # no gaps
            return False
        if not all(self._share):
            return True
        return any(has_offset(a) for ax in self for a in (ax.xaxis, ax.yaxis))

    def fit_axes_pad(self, pad):
        """Set the gaps to `pad` plus the room the tiles need between them, and the
        colorbar pad to `pad` plus what the tiles facing it draw towards it. Returns
        the inches the latter gained, as (horizontal, vertical), for the figure to
        absorb. Single pass: nothing is fitted again if the caller reformats the axes.
        """
        # gaps separate whole tiles, colorbars included; the colorbar pad separates a
        # tile from its own bar, so only the tile's own decorations count there
        over = np.array([[overhangs(ax) for ax in row] for row in self.axes_row])
        gaps = np.array([[tile_overhangs(ax) for ax in row] for row in self.axes_row])
        loc, all_ = self._colorbar_location, slice(None)
        side = SIDES.index(loc)
        # each gap must fit what the two tiles it separates reach into it
        h_pad = pad + (gaps[:, :-1, 2] + gaps[:, 1:, 0]).max(initial=0)
        v_pad = pad + (gaps[:-1, :, 1] + gaps[1:, :, 3]).max(initial=0)
        self.set_axes_pad((h_pad, v_pad))
        grown = np.zeros(2)
        if self._colorbar_mode:
            edges = {
                "left": (all_, 0),
                "right": (all_, -1),
                "top": (0,),
                "bottom": (-1,),
            }
            # every tile faces its own bar; otherwise only the edge row or column does
            faced = over if self._colorbar_mode == "each" else over[edges[loc]]
            grown[side % 2] = faced[..., side].max(initial=0)  # left/right are even
            self._cbar_pad_size.fixed_size += grown[side % 2]
        return grown
