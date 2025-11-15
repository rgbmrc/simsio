from collections.abc import Sequence
from typing import Self
from warnings import warn

import numpy as np
from matplotlib.scale import ScaleBase, scale_factory
from numpy.typing import NDArray

from simsio.utils import as_int

# region uniform grids


class LinearGrid:
    """1D grid of uniformly spaced points.

    At least one among `num` and `extent` must be given.

    Parameters
    ----------
    num
        Number of grid points or bins.
    extent
        Interval bounds [a, b] or length (b - a),
        converted to [a, b] using `origin` and `anchor`.
    step
        Grid step or spacing, namely bin width.
        Ignored if `num` and `extent` are given.
    anchor
        Anchoring of the grid, usually in the interval [0, 1].
        Used e.g. for rescaling and when `extent` is given as a scalar,
        specifying the relative position of `origin` inside [a, b].
        For example:
            anchor = 0.0 → origin = a
            anchor = 0.5 → origin = (a + b) / 2
            anchor = 1.0 → origin = b
        Default is 0.0.
    origin
        Reference point, used together with `anchor` when `extent`
        is given as a scalar length.
        Default is 0.0. Ignored if `extent` is given as [a, b].
    periodic
        If True, treat the grid as periodic (a ring); otherwise as an open
        segment. Periodicity affects e.g. how indices wrap in :meth:`index`.

    Exemples
    --------

    Easily reproduce numpy's arange and linspace, avoiding the common
    stop+=eps (arange) or num+=1 (linspace):

    >>> LinearGrid(2, step=0.5).edges  # arange(2+.01, step=0.5)
    >>> LinearGrid(2, num=4).edges  # linspace(0, 2, 4+1)

    """

    def __init__(
        self,
        extent: float | tuple[float, float] = None,
        num: int = None,
        *,
        step: float = None,
        anchor: float = 0.0,
        origin: float = None,
        periodic: bool = False,
    ):
        if num is None and extent is None:
            raise ValueError("At least one of 'num' or 'extent' must be given")
        if num is not None and extent is not None and step is not None:
            warn("'step' was passed but will be ignored")
        if np.size(extent) == 2 and origin is not None:
            warn("'origin' was passed but will be ignored")
        step = step or 1.0  # ignored if num and extent are given
        origin = origin or 0.0  # ignored if extent is given as interval
        extent = np.squeeze(step * num if extent is None else extent)
        if extent.size == 1:  # extent = (b - a)
            extent = origin + extent * (np.arange(2) - anchor)
        num = as_int(num or extent.ptp() / abs(step))
        assert num > 1 and extent.size == 2 and extent.ptp() != 0

        self.num = num
        self.extent = np.array(extent, dtype=float)  # copy
        self.step = self.extent.ptp() / self.num
        self.anchor = float(anchor)
        self.periodic = bool(periodic)

    @classmethod
    def from_points(cls, x: Sequence[float], periodic: bool = False) -> Self:
        """Construct a grid given evenly spaced data points.

        Parameters
        ----------
        x
            1D sequence of 2 or more evenly spaced grid points.
        periodic
            Periodicity. Default False.

        Returns
        -------
        Grid
            A Grid whose :meth:`points` reproduce the given `points`
            (within numerical precision).

        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size < 2:
            raise ValueError("points must be a 1D array with at least two elements")

        steps = np.diff(x)
        step = np.mean(steps)
        if not np.allclose(steps, step):
            raise ValueError(f"Grid points must be equally spaced; got steps {steps}")

        ext = (x[0] - 0.5 * step, x[-1] + 0.5 * step)
        return cls(ext, num=x.size, periodic=periodic)

    @property
    def points(self) -> NDArray:
        """Grid points (bin centers)

        `x_i = a + (i + 0.5) * step  for i = 0, ..., num-1`

        """
        return np.linspace(*(self.extent + self.step / 2), self.num, endpoint=False)

    @property
    def edges(self) -> NDArray:
        """Grid edges (bin boundaries)

        `e_i = a + i * step  for i = 0, ..., num`

        Same as `grid.dual(extremals=True).points`.

        """
        return np.linspace(*self.extent, self.num + 1)

    def index(self, vals: float | Sequence[float], weights=False) -> int | NDArray:
        """Return the index (or indices) for evaluating fields on the grid.

        For the non-periodic case, this identifies the subinterval containing
        `val`. For the periodic case, indices are wrapped modulo `num`.

        Parameters
        ----------
        vals
            Value(s) in data coordinates to be located on the grid.
        weights
            Placeholder for a future weighted interpolation API.
            Currently not implemented.

        Returns
        -------
        :
            If `weights` is False (the only supported mode):
            - For scalar `val`: a scalar integer index.
            - For array-like `val`: an array of indices with dtype int.

        Raises
        ------
        ValueError
            If the grid is non-periodic and any `val` lies strictly outside [a, b].
        NotImplementedError
            If `weights` is not None.

        """
        if weights:
            raise NotImplementedError("weights is not implemented yet")
            # TODO many rough edges, e.g. what if
            # not periodic and extent[0] < val < points[0]
            q, r = np.divmod(np.asanyarray(vals) - self.extent[0], self.step)
            q = q.astype(int) % self.num
            r /= self.step
            q = (q + np.rint(r)).astype(int)
            return ([q - 1, q], [r, r])

        x = np.asarray(vals, dtype=float)
        a, b = self.extent

        if not self.periodic and np.any(np.logical_or(x < a, x > b)):
            raise ValueError(
                f"{vals} is outside the grid interval {tuple(self.extent)}"
            )

        i = ((x - a) // self.step).astype(int)
        if self.periodic:
            i = i % self.num
        else:
            i = np.clip(i, None, self.num - 1)  # right endpoint val == b

        if np.isscalar(vals):
            return i.item()
        return i

    def block(self, factor: float) -> Self:
        """Coarsened grid corresponding to an RG blocking transformation.

        Parameters
        ----------
        factor
            Ratio between the original and the new number of points.
            E.g., factor=2 halves the resolution.

        Returns
        -------
        :
            New, blocked grid if factor != 1; otherwise the original grid.

        Notes
        -----
        The new number of points, computed as round(num / factor), must be
        an integer (up to numerical precision) greater or equal to 2.

        """
        if factor == 1:
            return self
        return type(self)(
            num=as_int(self.num / factor),
            extent=self.extent.copy(),
            periodic=self.periodic,
        )

    def dual(self, extremals=True) -> Self:
        """Dual grid whose points are the current partition boundaries.

        Parameters
        ----------
        extremals
            If True (default), include extremal boundaries, resulting
            in `num + 1` dual points.
            If False, use only interior boundaries, resulting in `num - 1`
            dual points.

        Returns
        -------
        :
            A new grid whose points (in data coordinates) coincide with the
            chosen boundaries of the original grid.

        Notes
        -----
        Currently periodic is passed through, in the future this may change.

        """
        # TODO handle periodicity, e.g. for building momentum space of a real space
        dual = type(self)(
            num=self.num - 1 + 2 * extremals,
            step=self.step,
            anchor=0.5,
            origin=self.extent.mean(),
            periodic=self.periodic,
        )
        dual.anchor = self.anchor  # reinstate anchor for further manipulation
        return dual

    def broadcast_to(self, num: int) -> Self:
        # TODO deprecate, this is not unique! e.g.
        # 3>2>8, 3>4>8, 3>9>8 are all valid
        # with larger numbers problems less likely (?)
        """Broadcasts the current grid to a new one with given 'num'.

        Broadcasting is achieved via a dual followed by block.

        """

        DeprecationWarning("Deprecated, not unique!")

        def _rgfactor(n_new, n_old):
            # returns n_old / n_new if n_old is a multiple or divisor of n_new, otherwise None
            return None if (n_old % n_new and n_new % n_old) else n_old / n_new

        fs = {(f, d) for d in (0, +1, -1) if (f := _rgfactor(num, self.num + d))}
        try:
            ((f, d),) = fs
        except ValueError as e:
            raise ValueError(
                f"Grid1D unbroadcastable from num={self.num} to num={num}"
            ) from e

        broadcasted = self
        if d != 0:
            broadcasted = broadcasted.dual(d > 0)
        broadcasted = broadcasted.rgflow(f)
        return broadcasted


ScaleLike = str | ScaleBase


def get_scale(scale: ScaleLike, **kwargs):
    if isinstance(scale, ScaleBase):
        return scale
    return scale_factory(scale, axis=None, **kwargs)


class UniformGrid:
    """1D grid of points x, uniformly spaced in a given scale.

    Uniform partition of an interval [a, b] in some *scaled* coordinate
    s = T(x), where T is the transform associated to a matplotlib scale.
    For scale="linear", it is equivalent to :class:`LinearGrid`.

    Parameters
    ----------
    lingrid
        A linear grid.
    scale
        Matplotlib scale name ("linear", "log", "symlog", ...),
        or a `ScaleBase` instance. Default is "linear".

    Notes
    -----
    Internally stores the :class:`LinearGrid` for s = T(x),
    T being the 1D transform associated with 'scale', and
    implements a translation layer between s and x.

    """

    def __init__(self, lingrid: LinearGrid, scale: ScaleLike):
        self.lingrid = lingrid
        self._init_scale_trans(scale)
        if not np.all(np.isfinite(self.extent)):
            raise ValueError(
                f"Linear grid extent {tuple(lingrid.extent)} is not "
                f"in the domain of the scale '{self.scale.name}'"
            )

    @classmethod
    def from_params(
        cls,
        extent: tuple[float, float],
        scale: ScaleLike,
        scale_extent: bool = True,
        **lingrid_kwds,
    ) -> Self:
        """Factory method mimicking :class:`LinearGrid`'s signature.

        If 'scale_extent' is True, 'extent' is taken to be in scaled coordinates and
        passed to :class:`LinearGrid` as-is, together with any keyword arguments.
        If 'scale_extent' is False (default), 'extent' is taken to be in data
        coordinates and converted to scaled coordinates internally. In this case,
        'extent' must be an interval [a, b] and 'origin' is ignored
        (a scalar extent only makes sense in linear scale).

        Exemples
        --------

        This factory method can reproduce e.g. both numpy's logspace and geomspace,
        while also allowing to use step instead of num.

        >>> UniformGrid.from_params([1e-2, 1e2], "log", step=0.5).edges  # geomspace
        >>> UniformGrid.from_params([-2, 2], "log", scaled=True, step=0.5).edges  # logspace

        """
        scale = get_scale(scale)
        if not scale_extent:
            assert np.size(extent) == 2, "Scalar extent not supported in data coords"
            extent = scale.get_transform().transform(extent)
        lingrid = LinearGrid(extent, **lingrid_kwds)
        return cls(lingrid, scale)

    @classmethod
    def from_points(
        cls, x: Sequence[float], scale: ScaleLike, periodic: bool = False
    ) -> Self:
        scale = get_scale(scale)
        s = scale.get_transform().transform(x)
        if not np.all(np.isfinite(s)):
            raise ValueError(
                "Points contain values outside the domain of the chosen scale"
            )
        return cls(LinearGrid.from_points(s, periodic), scale)

    @property
    def num(self):
        return self.lingrid.num

    @property
    def extent(self):
        return self._itransform(self.lingrid.extent)

    @property
    def points(self):
        return self._itransform(self.lingrid.points)

    @property
    def edges(self):
        return self._itransform(self.lingrid.edges)

    def index(self, val: float | NDArray) -> int | NDArray:
        return self.lingrid.index(self._transform(val))

    def block(self, factor: float) -> Self:
        return type(self)(self.lingrid.block(factor), self.scale)

    def dual(self, extremals: bool = True) -> Self:
        return type(self)(self.lingrid.dual(extremals), self.scale)

    def _init_scale_trans(self, scale):
        self.scale = get_scale(scale)
        trans = self.scale.get_transform()
        self._transform = trans.transform
        self._itransform = trans.inverted().transform


# endregion

# region irregular grids


def bin_edges(centers, pad=True):  # TODO check, bin_centers(edges)
    """Bin edges `e_i` from bin centers `x_i` for non-uniform grid.

    Such that `e_i = (x_i + x_{i-1}) / 2`?

    """
    if pad:
        start = 2 * centers[0] - centers[1]
        end = 2 * centers[-1] - centers[-2]
        centers = np.r_[start, centers, end]
    return (centers[1:] + centers[:-1]) / 2


# endregion
