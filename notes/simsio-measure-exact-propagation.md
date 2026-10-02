# safer Measure.exact propagation

Sonnet 5.5 plan.

## How `exact` leaks today

All the leaks come from one place: a Measure's attrs are copied wholesale, with `exact` treated like `cmap` or `label`.

| path | what happens to `exact` | verdict |
|---|---|---|
| `lazy_operator` with a scalar or unary op (`m*2`, `-m`, `m**2`) | attrs copied from `self` | wrong: the value changes, the exact doesn't |
| `lazy_operator` with a Function (`a - b`) | `compose_attrs(other, self)` takes the union, so `b.exact` leaks when `a` has none | wrong, and it silently picks one operand |
| comparison and boolean ops (`<`, `&`, `~`, `isin`) | same as above | meaningless: the result is a bool |
| `__getitem__` (`mass_gaps[0]`) | copied from `self`, so the index gets the full array | wrong. This is why `vector_gap` needs an explicit `exact=`. |
| `__matmul__` (`y @ abs_`, `y @ adev`) | copied from `self` | wrong for any non-identity filter |
| `__call__(**kwds)` (partial) | copied via `f_attrs` | dubious |
| `__pos__` (`+y`) | copied | harmless in principle |
| `Measure(y, scale="log")` and `register(_from=y)` | copied | intended: the same quantity with a different look |
| **`dev_exact` (my code)** | `Function(..., y, ...)` copies `y.exact`, so the deviation claims `exact = y.exact` | a bug I introduced |

## Design

`exact` is a parallel Measure that must undergo the same operation as its owner. That is what `filter` already is: `Measure.lazy_operator`, `__getitem__` and `__matmul__` each mirror the operation onto `self.filter`. So `exact` gets the same treatment.

1. **Stop the implicit inheritance.**
   - Add `NONINHERITED = {"exact"}` to `Function`.
   - `compose_attrs` drops those keys from both operands, so no composition path inherits `exact` any more.
   - `compose_attrs` already has a TODO about taking the operator. This is the minimum version of that TODO.
   - The `_from` copy in `Function.__init__` stays, so `Measure(y, scale="log")` and `register(_from=y)` keep `exact`.
2. **Propagate where it is valid, in `Measure` next to the `filter` code.**
   - Unary or scalar arithmetic (`neg`, `add`, `sub`, `mul`, `truediv`, `pow`, including the reflected forms): `res.exact = self.exact.lazy_operator(op, other, swap=swap)`.
   - Arithmetic between two Measures: both need an exact, then `res.exact = op(a.exact, b.exact)`.
   - Only one has an exact: result is `None`. The failure is loud and the user states the exact explicitly.
   - Comparison and boolean ops: `None`.
   - `__getitem__`: `res.exact = self.exact[index]`. Then `mass_gaps` can carry `exact=mass_gaps_exact`, `vector_gap` and `scalar_gap` get theirs for free, and the explicit `exact=` there goes away.
   - `__matmul__` with a Function `f`: `exact @ f` for elementwise or reduction filters. Which filters qualify I'd decide per filter, as an allowlist (see questions).
   - `__call__(**kwds)`: `None`, since the kwds may change what is measured.
3. **Parameters.**
   - `m`, `g`, `d` and `L` are exactly known, so `vector_gap - 2*m` should work.
   - I'd allow `exact=True`, meaning the value is its own reference. A helper `get_exact(f)` returns `f` if `f.exact is True`.
   - Set it on the parameter measures in `observables.py`.
   - This fits one of the "both operands need an exact" cases above, and it is optional.
4. **`dev_exact`.** Build the stack Function with `exact=None` explicitly, so deviations don't claim an exact. Raise a clear error when `y.exact` is None or missing.
5. **Docs and bookkeeping.**
   - Replace the "derived measures inherit" caveat in the simsio-analyze skill with the propagation rules.
   - Add a `lib/simsio/notes/backlog.md` entry for what stays deferred: units and scales, `norm`, and `default`, which have the same problem.
   - Commit in the submodule.

## Verification (no test suite exists, so a scratch script on real sims)

- `vector_gap.exact` equals `mass_gaps_exact[0]` through the `mass_gaps` indexing.
- `(vector_gap - 2*m).exact` evaluates to `M_V - 2m` once `m` has `exact=True`.
- `(vector_gap - e0).exact` is `None` when `e0` has no exact, and `(vector_gap - 2*m)` without `m`'s `exact=True` is also `None`.
- `(y < 1).exact` is `None`, and `dev_exact(y).exact` is `None`.
- `Measure(y, scale="log").exact is y.exact`.
- `report_1d` with `dev_exact` still plots.

## Questions

- Do you want the `exact=True` mechanism for parameters (item 3)? It makes `gap - 2*m` work, and without it that case is `None`.
- For `@` filters, should I propagate to `exact` by default and opt out on the dev, stack and round filters? Or opt in with a flag on the Function? Opt-out is the more convenient default. Opt-in is safer, which matches your "ticking bomb" concern. I'd pick opt-in and mark `re_`, `abs_`, `sqrt_`, `mean_`, `max_` and `sum_`.
