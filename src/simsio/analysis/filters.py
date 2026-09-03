import mplotter as plotter  # DEL
import numpy as np
from matplotlib import colors, ticker

from simsio.analysis.numpy_extras import fftsymshift
from simsio.analysis.quantities import Function, Measure, id_, nomath

__all__ = [  # noqa: RUF022
    "id_",
    # basic
    "re_",
    "im_",
    "re_and_im",
    "assert_real",
    "abs_",
    "arg_",
    "abs_and_arg",
    "max_",
    "sum_",
    "sqrt_",
    "mean_",
    "round_",
    "all_",
    "any_",
    # linalg
    "norm_",
    # abs & rel deviation
    "adev",
    "rdev",
    "adev_log",
    "rdev_log",
    # fourier
    "fourier",
]

re_or_im_attrs = {"cmap": "bwr_r", "norm": colors.CenteredNorm()}
dev_attrs = {"cmap": "RdBu", "norm": colors.CenteredNorm()}
arg_attrs = {
    "cmap": "twilight",
    "norm": colors.Normalize(-np.pi, +np.pi),
    "cbar_kwds": {
        "ticks": ticker.MultipleLocator(np.pi),
        "format": plotter.annotating.SSFractionFormatter(1, unit=(np.pi, r"\pi")),
    },
}

re_ = Function(np.real, label=r"$\Re$", **re_or_im_attrs)
im_ = Function(np.imag, label=r"$\Im$", **re_or_im_attrs)
abs_ = Function(np.abs, label=r"$|${x}$|$".format, cmap="viridis", norm=None)
arg_ = Function(np.angle, label=r"$\arg$", **arg_attrs)
conj_ = Function(np.conj, label=r"{x}$^*$".format)
max_ = Function(np.max, label=r"$\max$")
sum_ = Function(np.sum, label=r"$\sum$")
# FIXME nomath in label breaks text
sqrt_ = Function(np.sqrt, label=lambda x: rf"$\sqrt{{{nomath(x)}}}$")
mean_ = Function(np.mean, label=lambda x: rf"$\overline{{{nomath(x)}}}$")
round_ = Function(np.round, label=id_)
all_ = Function(np.all)
any_ = Function(np.any)


@Function.register(label=id_)
def assert_real(a):
    a = np.asanyarray(a)
    assert np.allclose(a.imag, 0)
    return a.real


def abs_and_arg(measure: Measure) -> Measure:
    return measure @ abs_, measure @ arg_


def re_and_im(measure: Measure) -> Measure:
    return measure @ re_, measure @ im_


@Function.register(label=r"$\|${x}$\|$".format, cmap="viridis", norm=None)
def norm_(x, *, ndim=None, axis=None, ord=2):
    # OPT optimize a little bit
    if ord == 1:
        y = abs(x)
    else:
        y = x * np.conj(x)  # for ord == 2 or None this is enough
    if ord not in {1, 2}:
        y **= ord / 2
    if axis is None and ndim is not None:
        axis = tuple(range(-1, -ndim - 1, -1))
    return np.sum(y, axis=axis) ** (1 / ord)


def get_dev_operands(x):
    *y, ref = x
    return y[0] if len(y) == 1 else y, ref


@Function.register(label=r"{x}$\text{{ dev.}}$".format, **dev_attrs)
def adev(x):
    y, ref = get_dev_operands(x)
    return y - ref


adev_log = Measure(adev, scale="log")


@Function.register(label=r"{x}$\text{{ rel. dev.}}$".format, **dev_attrs)
def rdev(x):
    y, ref = get_dev_operands(x)
    return y / np.abs(ref) - np.sign(ref)


rdev_log = Measure(rdev, scale="log")


@Function.register(label=r"$\mathcal{F}$")
def fourier(dat, axis=None, norm="ortho"):
    s = np.array(dat.shape)
    if axis is not None:
        s = s[np.asarray(axis)]
    s += s % 2
    return fftsymshift(np.fft.fftn(dat, s=s, axes=axis, norm=norm), axis=axis)
