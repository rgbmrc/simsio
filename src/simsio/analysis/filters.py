import numpy as np
from matplotlib import colors, ticker
import cmcrameri as ccm  # DEL
import mplotter as plotter  # DEL

from .quantitites import Function, id_, nomath

__all__ = [
    "id_"
    # np.*
    "re_",
    "im_",
    "abs_",
    "arg_",
    "max_",
    "sum_",
    "sqrt_",
    "mean_",
    "round_",
    "all_",
    "any_",
    # np.linalg.*
    "norm_",
    # abs & rel deviation
    "adev",
    "rdev",
]

re_or_im_attrs = {"cmap": ccm.vik_r, "norm": colors.CenteredNorm()}
arg_attrs = {
    "cmap": "twilight_shifted_r",
    "norm": colors.Normalize(-np.pi, +np.pi),
    "cbar_kwds": {
        "ticks": ticker.MultipleLocator(np.pi),
        "format": plotter.annotating.SSFractionFormatter(1, unit=(np.pi, "\pi")),
    },
}
re_ = Function(np.real, label=r"$\Re$", **re_or_im_attrs)
im_ = Function(np.imag, label=r"$\Im$", **re_or_im_attrs)
abs_ = Function(np.abs, label=r"$|${x}$|$".format, cmap="viridis", norm=None)
arg_ = Function(np.angle, label=r"$\arg$", **arg_attrs)
max_ = Function(np.max, label=r"$\max$")
sum_ = Function(np.sum, label=r"$\sum$")
# FIXME nomath in label breaks text
sqrt_ = Function(np.sqrt, label=lambda x: rf"$\sqrt{{{nomath(x)}}}$")
mean_ = Function(np.mean, label=lambda x: rf"$\overline{{{nomath(x)}}}$")
round_ = Function(np.round, label=id_)
all_ = Function(np.all)
any_ = Function(np.any)


@Function.register(label="$\|${x}$\|$".format, cmap="viridis", norm=None)
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


@Function.register(
    label=r"{x}$\text{{ dev.}}$".format, cmap="RdBu", norm=colors.CenteredNorm()
)
def adev(x):
    y, ref = get_dev_operands(x)
    return y - ref


@Function.register(adev, label=r"{x}$\text{{ rel. dev.}}$".format)
def rdev(x):
    y, ref = get_dev_operands(x)
    return y / np.abs(ref) - np.sign(ref)
