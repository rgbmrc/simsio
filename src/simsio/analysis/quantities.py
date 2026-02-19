# ruff: noqa: E731 # lambdas are convenient here

import inspect
import operator
import re
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Callable, Self

import dpath
import matplotlib as mpl
import numpy as np
import xarray as xr

from simsio.analysis.utils import is_numeric, as_ndarray
from simsio.simulations import get_sim, purge_caches
from simsio.utils import as_scalar

__all__ = ["Function", "Measure", "id_", "indices_to_str"]


def math_str(val):
    return f"${str(val)}$" if is_numeric(val) else str(val)


def nomath(text):
    return text.replace("$", "")


USE_TEX = True  # read matplotlib rc settings? simsio settings?
_DEFAULT_SENTINEL = ...  # object() or dpath._DEFAULT_SENTINEL unstable, why?
_NO_ARG_SENTINEL = object()
_FUNC_ARG = r"$\:\cdot\:$"
_OP_REGEX = re.compile(r"Same as (\W*(a|b)\W*?(a|b)?\W*)\.")
_CUSTOM_OP_LABEL_TEX = {  # only for label, not name
    operator.pow: r"{a}$^${b}",
    operator.mul: r"{a}{l}\cdot{r}{b}",
}


def closest_common_ancestor(*cls_list):
    mros = [list(inspect.getmro(cls)) for cls in cls_list]
    track = defaultdict(int)
    while mros:
        for mro in mros:
            cur = mro.pop(0)
            track[cur] += 1
            if track[cur] == len(cls_list):
                return cur
            if len(mro) == 0:
                mros.remove(mro)
    return None  # or raise, if that's more appropriate


def get_operator_symbol(op) -> str:
    # NOTE make this a static method of Function?
    # for op in dir(operator):
    #     if not op.startswith("_"):
    #         print(getattr(operator, op).__doc__.splitlines()[0])
    if isinstance(op, str):
        return op
    if m := _OP_REGEX.fullmatch(op.__doc__):
        l = m[1]
        l = l.replace("a", "{a}").replace("b", "{b}")
        if l.count(" ") == 2:  # generalize to any even number of spaces?
            l = l.replace(" ", "{l}", 1).replace(" ", "{r}", 1)
        return l
    return "{a}__" + getattr(op, "__name__", str(op)) + "__{b}"


def indices_to_str(inds) -> str:
    if inds == Ellipsis:
        return "..."
    if inds == np.newaxis:
        return "newaxis"
    if isinstance(inds, slice):
        t = (inds.start, inds.stop, inds.step)
        return "".join(str(c) if c is not None else ":" for c in t)
    try:
        return ",".join(indices_to_str(ind) for ind in inds)
    except TypeError:
        return str(inds)


class Function:
    _register = {}
    INVALID_NAMES = {None, "<lambda>", "None"}
    MERGED_ATTRS = {"line", "image", "axis", "cbar", "legend"}

    def __init__(self, func: Callable, _from: dict | Self = None, **attrs):
        if not callable(func):
            raise TypeError(f"{func} is not callable")
        if _from:
            _from_attrs = deepcopy(getattr(_from, "attrs", _from))
            _from_attrs.pop("func", None)
            _from_attrs.pop("name", None)
            attrs = _from_attrs | attrs
        f_wrapped = f_attrs = func
        if isinstance(func, partial):
            f_wrapped = f_attrs = func.func
        if isinstance(f_attrs, Function):
            _f_attrs_attrs = deepcopy(f_attrs.attrs)
            _f_attrs_attrs.pop("func", None)
            _f_attrs_attrs.pop("name", None)
            attrs = _f_attrs_attrs | attrs
        if type(func) is type(self):  # TODO too strict?
            # if isinstance(self, type(f_attrs)):  # NOTE not the reverse!
            func = f_attrs.func
            # FIXME also preserve name by default
        if isinstance(func, partial):
            partial_args = (f"{v}" for v in func.args)
            partial_kwds = (f"{k}={v}" for k, v in func.keywords.items())
            partial_str = ",".join(chain(partial_args, partial_kwds))
            name = self.compose_names(f_attrs, "{a}({b})", partial_str)
            attrs.setdefault("name", name)
            func = partial(f_wrapped, *func.args, **func.keywords)
        # 1. name (need final value for later, hence attrs query)
        self.name = attrs.get("name", f_attrs.__name__)
        # 2. default values
        self.label = self.name
        self.default = _DEFAULT_SENTINEL
        # NOTE consider this, then base and filter in Measure become (lazy?) properties
        #      lazy_operator can be implemented using id_ (identity) or as a class, defining
        #      Function @ LazyOperator (avoiding the eager evaluation of id_.lazy_operator)
        # self.components = [self]  # segments, parts?
        # 3. overwritten by (includes __name__ == name)
        # update_wrapper(self, f_attrs)
        # 4. overwritten by (includes deepcopy of f_attrs.attrs)
        for k, v in attrs.items():
            setattr(self, k, v)
        # 5. override everything
        self.func = func
        if isinstance(self.label, Function):
            self.label = self.label.func  # avoid "partial"
        elif not callable(self.label) and USE_TEX and not self.label.startswith("$"):
            self.label = rf"$\text{{{self.label}}}$"
        assert self.name not in self.INVALID_NAMES
        if self.name in self._register:
            registered_self = self._register[self.name]
            if (
                type(registered_self) is not type(self)
                or self.func is not registered_self.func
            ):
                # TODO output a sensible amount of warnings
                # warn(f"Overwriting {self.name} with {self!r}")
                pass
        self._register[self.name] = self

    @property
    def attrs(self) -> dict:
        # OPT use functools.WRAPPER_UPDATES
        return {
            k: v for k, v in vars(self).items() if not k.startswith("_") and k != "func"
        }

    def setdefault(self, attr, value):
        try:
            return getattr(self, attr)
        except AttributeError:
            setattr(self, attr, value)
            return value

    @property
    def __name__(self) -> str:
        return self.name

    @__name__.setter
    def __name__(self, value):
        self.name = value

    def _repr_latex_(self) -> str:
        # https://ipython.readthedocs.io/en/stable/config/integrating.html
        # TODO USE_TEX
        return self.string()

    def __repr__(self) -> str:
        return f"{type(self).__name__}:{self.name}"

    def __str__(self) -> str:
        return self.string()

    def __hash__(self) -> int:
        # return hash(self.name) # NOTE for xarray, unsafe
        return hash(repr(self))

    def __eq__(self, other) -> bool:
        # return self.name == getattr(other, "name", other) # NOTE for xarray, unsafe
        # NOTE dangerous!
        return repr(self) == repr(other)

    # should we allow for more args or kwds,
    # register & overloaded operators need review
    def __call__(self, *args, **kwds):
        if not args:
            return type(self)(partial(self, **kwds))
        try:
            return self.func(*args, **kwds)
        except Exception as e:
            # FIXME we used to return default (if any) but now default is Measure-only
            # TODO improve formatting
            raise ValueError(f"Error computing {self!r} on {args}, {kwds}") from e

    def __matmul__(self, other):
        other = Function.from_callable(other)
        attrs = self.compose_attrs(self, other)
        if other.default is _DEFAULT_SENTINEL and self.default is not _DEFAULT_SENTINEL:
            try:
                attrs["default"] = other(self.default)
            except ValueError:
                pass
        if not other.label:  # avoids unnecessary spacing
            label = self.label
        elif callable(other.label):
            if callable(self.label):
                label = lambda x: other.label(x=self.label(x=x))
            else:
                label = other.label(x=self.label)
        else:
            label = self.compose_labels(self, r"{b}$\:${a}", other)
        return type(self)(
            lambda x: other.func(self.func(x)),
            attrs,
            name=self.compose_names(self, operator.matmul, other),
            label=label,
            # components=self.components + [other], # NOTE __init__
        )

    def __rmatmul__(self, other):
        return Function.from_callable(other) @ self

    @staticmethod
    def compose_names(a, op, b=None) -> str:
        a, b = (getattr(x, "__name__", str(x)) for x in (a, b))
        name = get_operator_symbol(op).format(a=a, b=b, l="", r="")
        return f"({name})"  # TODO avoid unnecessary parentheses

    @classmethod
    def compose_labels(cls, a, op, b=None) -> str | Callable:
        a, b = (getattr(x, "label", math_str(x)) for x in (a, b))
        # TODO empty (not id_) labels produce weird results, try `uuid + uuid`
        if callable(a) and callable(b):
            return lambda x: cls.compose_labels(a(x=x), op, b(x=x))
        elif callable(a):
            return lambda x: cls.compose_labels(a(x=x), op, b)
        elif callable(b):
            return lambda x: cls.compose_labels(a, op, b(x=x))
        return get_operator_symbol(op).format(a=a, b=b, l="${}", r="{}$")

    @classmethod
    def compose_attrs(cls, a: dict | Self, b: dict | Self) -> dict:
        # TODO take operator as input and handle norm, default, ..., then:
        # - in lazy_operator, call this also when other is not a Function
        # - in __matmul__, op should correspond to calling other
        # OPT avoid redundant deepcopies, do not copy dicts, check also __init__
        a, b = (deepcopy(getattr(x, "attrs", x)) for x in (a, b))
        attrs = a
        for k, v in b.items():
            if k in cls.MERGED_ATTRS:
                attrs[k] |= v
            else:
                attrs[k] = v
        return attrs

    def lazy_operator(self, op, other=None, *, op_fmt=None, swap=False) -> Self:
        # ancestor = closest_common_ancestor(type(self), type(other))
        # if not issubclass(ancestor, Function):
        #     ancestor = Function
        cls = type(self)  # store before eventual swap
        try:
            other = type(self).from_callable(other)
        except TypeError:
            attrs = self
            if other is None:
                func = lambda x: op(self.func(x))
            elif swap:
                # swap before lambda to avoid late binding closure issues
                self, other = other, self
                func = lambda x: op(self, other.func(x))
            else:
                func = lambda x: op(self.func(x), other)
        else:
            attrs = cls.compose_attrs(other, self)  # self takes precedence
            func = lambda x: op(self.func(x), other.func(x))
        name = cls.compose_names(self, op, other)
        op_fmt = op_fmt or _CUSTOM_OP_LABEL_TEX.get(op) or op
        label = cls.compose_labels(self, op_fmt, other)
        return cls(func, attrs, name=name, label=label)

    def __neg__(self):
        return self.lazy_operator(operator.neg)

    def __add__(self, other):
        return self.lazy_operator(operator.add, other)

    def __sub__(self, other):
        return self.lazy_operator(operator.sub, other)

    def __mul__(self, other):
        return self.lazy_operator(operator.mul, other)

    def __truediv__(self, other):
        return self.lazy_operator(operator.truediv, other)

    def __pow__(self, other):
        return self.lazy_operator(operator.pow, other)

    # reflected binary operators

    def __rtruediv__(self, other):
        return self.lazy_operator(operator.truediv, other, swap=True)

    def __rpow__(self, other):
        return self.lazy_operator(operator.pow, other, swap=True)

    # assume following operators are commutative on codomain
    # attrs precedence is not a problem because other is not a Function

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    # comparison operators (==, != not overloaded)

    def equal(self, other):
        return self.lazy_operator(operator.eq, other)

    def isin(self, other):
        return self.lazy_operator(operator.contains, other, swap=True)

    def __lt__(self, other):
        return self.lazy_operator(operator.lt, other)

    def __le__(self, other):
        return self.lazy_operator(operator.le, other)

    def __gt__(self, other):
        return self.lazy_operator(operator.gt, other)

    def __ge__(self, other):
        return self.lazy_operator(operator.ge, other)

    # boolean operators

    def __invert__(self):
        return self.lazy_operator(operator.inv)

    def __and__(self, other):
        return self.lazy_operator(operator.and_, other)

    def __or__(self, other):
        return self.lazy_operator(operator.or_, other)

    # iteration and indexing

    def __getitem__(self: Self, index) -> Self:
        if callable(index):
            return NotImplemented
        return type(self)(
            # passing kwds through makes sense in this case
            lambda x, **kwds: self.func(x, **kwds)[index],
            self,
            name=self.compose_names(self, operator.getitem, indices_to_str(index)),
            label=self.label,
        )

    def __iter__(self):
        raise TypeError("Function is not iterable")

    @classmethod
    def register(cls, _from=None, **attrs) -> Self:
        def _register_func(func):
            # even if func is already an instance of cls,
            # a new instance allows to overwrite attrs
            # TODO but it must have different name?
            func = cls(func, _from, **attrs)
            name = func.name
            # Function._register[name] = func
            if hasattr(func, "cmap") and not isinstance(func.cmap, str):
                mpl.colormaps.register(func.cmap, name=name, force=True)
            return func

        return _register_func

    @classmethod
    def from_callable(cls, func):
        """Pass subclasses through."""
        if isinstance(func, cls):
            return func
        cls_wrapped = type(getattr(func, "func", None))
        if issubclass(cls_wrapped, cls):
            cls = cls_wrapped
        return cls(func)

    @classmethod
    def from_path(cls, path, **kwds):
        kwds.setdefault("name", path)
        default = kwds.get("default", _DEFAULT_SENTINEL)
        func = partial(dpath.get, glob=path, default=default)
        return cls(func, **kwds)

    @classmethod
    def get(cls, func_like):
        if func_like is None:  # TODO same for masked/nan?
            return
        if isinstance(func_like, str):
            try:
                return cls._register[func_like]
            except KeyError:
                return cls.from_path(func_like)  # TODO raise? optional behavior?
        return cls.from_callable(func_like)

    @classmethod
    def get_array(cls, func_like):
        # OPT avoid np.frompyfunc every time? mah, it's fast
        return np.frompyfunc(cls.get, 1, 1)(func_like)

    def string(self, x=None, val=_DEFAULT_SENTINEL, fmt="", sep=None, math=None):
        # TODO USE_TEX
        l = self.label
        if callable(l):
            l = l(x=_FUNC_ARG)
        if val is _DEFAULT_SENTINEL and x is not None:
            val = self(x)
        if val is not _DEFAULT_SENTINEL:
            val = as_scalar(val)
            if sep is None:
                sep = "${}={}$" if l else ""
            if math is None:
                math = is_numeric(val)
            if math:  # str since np.nan != np.nan and np.nan is not float("nan")
                match re.fullmatch("([+-]?)(.*)", str(val)).groups():
                    case (s, "inf"):
                        val = s + r"\infty"
                        fmt = "s"
                    case ("", "nan"):
                        val = r"\mathrm{NaN}"
                        fmt = "s"
            m = "$" if math else ""
            l = f"{l}{sep}{m}{val:{fmt}}{m}"
        return l.replace("$$", "")  # OPT could also remove $${} & {}$$?

    @classmethod
    def strings(cls, funcs, vals, junc=";"):
        return junc.join(f.string(val=v) for f, v in zip(funcs, vals))


class Measure(Function):
    def __init__(self, func, _from=None, **attrs):
        self.cached = True
        self.filter = None
        # TODO document .base is not perfect and only works with @:
        # measure**2 has itself as base, not measure (with filter=lambda x: x**2)
        # override _from.base (should we leave it?)
        attrs.setdefault("base", self)
        registered_self = self._register.get(self.name, None)
        if registered_self is not None and self.func is not registered_self.func:
            purge_caches([registered_self])
        super().__init__(func, _from, **attrs)

    # def __init__(
    #     self, func, name=None, label=None, otypes=None, signature=None, **attrs
    # ):
    #     # TODO: implement cahcing, requires key or similar,
    #     # so it should be done at the Measure level,
    #     # not for a generic function, uness we use id()
    #     # simultaneously, it must be done by the function
    #     # passed to vectorized ...
    #     super().__init__(func, key, label, **attrs)  # set __name__
    #     func = sim_like_arg(cached(func, self.__name__))  # enable chaching
    #     super().__init__(func, key, label)  # update func
    #     self._vect = np.vectorize(self.func, otypes=otypes, signature=signature)

    # def __getattribute__(self, name: str):
    #     try:
    #         super().__getattribute__(name)
    #     except AttributeError:
    #         return self._attrs[name]

    # def __setattr__(self, name: str, value) -> None:
    #     print(name)
    #     super().__setattr__(name, value)

    def __call__(self, sims_like=_NO_ARG_SENTINEL, **kwds):
        # should we allow args/kwds result caching and
        # register keys need to be reviewed
        if sims_like is _NO_ARG_SENTINEL:
            return super().__call__(**kwds)  # partial
        try:
            sims_like = get_sim(sims_like)
        except (ValueError, TypeError):  # e.g. ndarray
            return self.vectorized(sims_like, **kwds)
        if not sims_like:
            if sims_like is np.ma.masked or self.default is _DEFAULT_SENTINEL:
                return np.ma.masked
            return self.default
        if self.cached and not kwds:  # FIXME what? LGTM
            if self not in sims_like.cache:
                sims_like.cache[self] = super().__call__(sims_like)
            return sims_like.cache[self]
        else:
            return super().__call__(sims_like, **kwds)

    def vectorized(self, sims, **kwds):
        # np.vectorize does not play well with masks & arbitrary otypes
        # and is more general than needed, so we implement this ourselves
        sims_array = as_ndarray(sims)
        if not sims_array.shape:
            raise TypeError(f"Error computing {self!r}.vectorized on {sims}")
        # get_sim first to detect missing sims (e.g., nan evaluates to True)
        sims_array = np.frompyfunc(get_sim, 1, 1)(sims_array)  # respects mask
        # guess output dtype and shape
        # skip missing sims (masked/default value) to ensure homogeneous
        # even *if* we wanted to broadcast, np.broadcast_arrays ignores mask
        # in principle we could infer shape & dtype from first valid sim
        # but, e.g., numpy casts float to int silently truncating decimals
        # while the following converts ints to floats & fails for scalars
        # dtype = np.common_type(*out); shape = np.broadcast(*out).shape
        out = np.ma.array([self(sim, **kwds) for sim in sims_array.flat if sim])
        shape = sims_array.shape + out.shape[1:]
        try:
            out = out.reshape(shape)
        except ValueError:  # there were missing sims
            out, out_it = np.ma.masked_all(shape, out.dtype), iter(out)
            for ij, sim in np.ndenumerate(sims_array):
                # even if get_sim() -> None, call self for consistency
                # with non-vectorized (e.g., regarding masked/default)
                out[ij] = next(out_it) if sim else self(sim, **kwds)
        else:
            # if sims is a masked array, we preserve the mask, even if trivial
            if not np.ma.is_masked(out) and not isinstance(sims, np.ma.MaskedArray):
                out = out.data
        return out

    def xarray(self, sims: xr.DataArray, **kwds):
        dat = self.vectorized(sims, **kwds)
        dat_dims = tuple(map((self.name + "[{}]").format, range(dat.ndim - sims.ndim)))
        dat_dims = tuple(map(f"{self.name}[{{}}]".format, range(dat.ndim - sims.ndim)))
        dat_dims = tuple(f"{self.name}[{i}]" for i in range(dat.ndim - sims.ndim))
        return xr.DataArray(dat, sims.coords, sims.dims + dat_dims)

    def lazy_operator(self, op, other=None, *, op_fmt=None, swap=False):
        res = super().lazy_operator(op, other, op_fmt=op_fmt, swap=swap)
        if self.filter:
            res.filter = self.filter.lazy_operator(op, other, op_fmt=op_fmt, swap=swap)
        return res

    def __getitem__(self: Self, index) -> Self:
        res = super().__getitem__(index)
        if self.filter:
            res.filter = self.filter[index]
        return res

    def __matmul__(self, other):
        if isinstance(other, Measure):
            # compose with other.__call__
            other = +other
        comp = super().__matmul__(other)
        comp.base = self.base
        if f := self.filter:
            comp.filter = other if f == id_ else f @ other

        return comp

    def __pos__(self):
        # TODO disable filter?
        return Function(self, name=self.compose_names(self, operator.pos))

    def string(self, sim=None, *, val=_DEFAULT_SENTINEL, fmt="", sep=None, math=None):
        if val is _DEFAULT_SENTINEL and sim is not None:
            val = self(sim)
        return super().string(val=val, fmt=fmt, sep=sep, math=math)

    @classmethod
    def strings(cls, funcs, sim=None, *, vals=_DEFAULT_SENTINEL, junc=";"):
        if vals is _DEFAULT_SENTINEL and sim is not None:
            vals = [f(sim) for f in funcs]
        return super().strings(funcs, vals, junc=junc)


id_ = Function(lambda x: x, name="id", label="{x}".format)
