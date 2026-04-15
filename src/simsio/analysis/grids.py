from math import isclose

import numpy as np


def bin_edges(centers, pad=True):
    if pad:
        start = 2 * centers[0] - centers[1]
        end = 2 * centers[-1] - centers[-2]
        centers = np.r_[start, centers, end]
    return (centers[1:] + centers[:-1]) / 2


def _float_as_int(fval):
    ival = round(fval)
    if not isclose(ival, fval):
        raise ValueError()
    return ival


class Grid:
    pass


class LinGrid(Grid):
    """1-dimensional grid of evenly spaced points.

    Uniform partition of an interval [a, b] in n subintervals.
    Each grid point corresponds to the center of one subinterval.

    The values a, b can be given explicitely, or via (b - a) and an anchoring point specification.

    Parameters
    ----------
    n : int, optional (at least one among `n` and `extent` must be given)
        Number of points/subintervals, by default `round((b - a) / step)`.
    extent : (tuple[float] | float), optional (at least one among `n` and `extent` must be given)
        Interval [a, b]; or its size (b - a), converted to [a, b] via `origin` and `anchor`,
        by default `step * n`.
    step : float, optional (ignored if both `n` and `extent` are provided)
        Spacing between the grid points, by default `1.0`.
    origin : float, optional (ignored if `extent` is given as [a, b])
        See `anchor`, by default `0.0`.
    anchor : float, optional (ignored if `extent` is given as [a, b])
        Relative location of `origin` on the interval, by default `0.0`.
    periodic : bool, optional
        Wether the topology of the interval is
        an open segment (False, default)
        or a closed ring (True).

    Attributes
    ----------
    n : int
        Number of grid points/subintervals.
    step : float
        Spacing between the points, i.e. length of the subintervals, (b - a) / n.
    extent : ndarray
        Array [a, b] of the lower and upper extramals of the interval.
    periodic : bool
        Topology of the interval: open segment or closed ring.

    """

    def __init__(
        self,
        n=None,
        extent=None,
        step=None,
        origin=None,
        anchor=None,
        periodic=False,
    ):
        step = step or 1.0  # ignored if n and extent are given
        extent = np.asarray(extent if extent is not None else step * n, dtype=float)
        # interval size -> interval extremals, via origin and anchor
        if extent.size == 1:
            extent = (origin or 0.0) + extent * (np.linspace(0, 1, 2) - (anchor or 0.0))
        assert extent[0] < extent[1]
        self.n = _float_as_int(n or np.ptp(extent) / step)
        self.step = np.ptp(extent) / self.n
        self.extent = extent
        self.periodic = bool(periodic)

    @classmethod
    def from_points(cls, points):
        steps = np.diff(points)
        step = np.mean(steps)
        if not np.allclose(steps, step):
            raise ValueError(f"Grid points must be equally spaced, recieved\n{points}")
        assert step > 0
        return cls(
            n=len(points),
            extent=[points[0] - 0.5 * step, points[-1] + 0.5 * step],
        )

    def points(self):
        """Returns the grid points.

        Returns
        -------
        numpy.ndarray
            Grid points, namely centers of the subintervals of the partition.

        """
        return np.linspace(*(self.extent + self.step / 2), self.n, endpoint=False)

    def index(self, val, weights=False):
        """Returns the index(es) for evaluating fields living on the grid.

        If the grid is periodic, indices are treated modulo n;
        otherwise ValueError is raised if val is outside the interval.

        Parameters
        ----------
        val : float
            Value(s) to be located on the grid.
        weights : bool, optional
            Wether to compute the data for a simple evaluation (False, default)
            or for a weighted mean (True).

        Returns
        -------
        int or (list[int], list[float])
            * index `i` of the grid point closest to val (index of the subinterval containing val); or
            * indices `is` of the two closest points to val & weights `ws` such that:
            `numpy.inner(points()[is], ws) == val`.

        """
        if (
            not self.periodic
            and np.logical_or(val < self.extent[0], val > self.extent[1]).all()
        ):
            raise ValueError(f"{val} is outside the grid interval {tuple(self.extent)}")
        if weights:
            raise NotImplementedError
            # TODO: many probles below + what if
            # not periodic and extent[0] < val < points()[0]
            # or analogous @ right end?
            q, r = np.divmod(np.asanyarray(val) - self.extent[0], self.step)
            q = q.astype(int) % self.n
            r /= self.step
            q = (q + np.rint(r)).astype(int)
            return ([q - 1, q], [r, r])
        else:
            return ((np.asanyarray(val) - self.extent[0]) // self.step).astype(
                int,
            ) % self.n

    def rgflow(self, factor):
        """Builds a grid corresponding to the renormalization group flow of
        the current one.

        Parameters
        ----------
        factor : float
            Coarse graining factor:
            original num of points / num of points after the RG transformation.

        Returns
        -------
        Grid1D
            A new grid corresponding to the RG transformation of the current one.

        """
        if factor == 1:
            return self
        return Grid1D(n=round(self.n / factor), extent=self.extent)

    def dual(self, extremals=True):
        """Builds a dual grid, whose points are the current partition bounds.

        Extremals can be discarded.

        Parameters
        ----------
        extremals : bool, optional
            Whether the extremals of the original interval should be included, by default True.

        Returns
        -------
        Grid1D
            A new grid, whose points are the bounds of the subintervals of the current grid.

        """
        # TODO: handle periodicity, e.g. for building momentum space of a real space
        return Grid1D(
            n=self.n - 1 + 2 * extremals,
            step=self.step,
            origin=self.extent.mean(),
            anchor=0.5,
        )

    def broadcast_to(self, n):
        # TODO: this is not unique! e.g.:
        # 3>2>8, 3>4>8, 3>9>8 are all valid
        # with larger numbers problems less likely (?)
        """Broadcasts the current grid to a new one with the given number of
        points.

        Broadcasting is achieved via a duality and an eventual subsequent RG transformation.

        Parameters
        ----------
        n : int
            Nnumber of points of the broadcasted grid

        Returns
        -------
        Grid1D
            A new grid, with n points, obtained broadcasting the current one.

        """

        def _rgfactor(n_new, n_old):
            # returns n_old / n_new if n_old is a multiple or divisor of n_new, otherwise None
            return None if (n_old % n_new and n_new % n_old) else n_old / n_new

        fs = {(f, d) for d in (0, +1, -1) if (f := _rgfactor(n, self.n + d))}
        try:
            ((f, d),) = fs
        except ValueError as e:
            raise ValueError(f"Grid1D unbroadcastable from n={self.n} to n={n}") from e

        broadcasted = self
        if d != 0:
            broadcasted = broadcasted.dual(d > 0)
        broadcasted = broadcasted.rgflow(f)
        return broadcasted


Grid1D = LinGrid


class LogGrid(Grid):
    pass
