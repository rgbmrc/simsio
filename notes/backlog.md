# Backlog

Persistent memory of unfinished business. Append freely; move items to a changelog
once the library is mature enough for one. Absorbs the former `ideas.md`.

Items are grouped **by area** (mirroring the maturity tiers in `CLAUDE.md`) and
prefixed by a marker, so that severity survives the grouping:

| marker | meaning |
|---|---|
| 🐛 | bug: wrong results, data loss, or a crash on a supported path |
| 🩹 | rough edge: works, but wrong or awkward in some cases |
| 🚧 | half-implemented: started and left in the middle |
| 💡 | idea: not started, may never be |
| ✅ | settled, kept only as context for a neighbouring item |

Same classification as the commit gitmoji, so an item and the commit closing it
usually carry the same marker.

## core

`settings`, `configs`, `iocore`, `serializers`, `simulations`, `runsim`.

- 🐛 `configs.cfg_pop` iterates `Path(dir).glob(u)` where `dir` is the builtin, and
  the glob ignores the handler path template it just built.
- 🐛 `configs.cfg_sort` raises `NotImplementedError` on entry; the body references
  `uids_sort`, which cannot be imported without a circular import
  (`configs` ← `analysis.organize`).
- 🚧 `configs._config_path_history` should be backed by `HISTORY_FILE`
  (`.simsio_history`), currently an in-process deque only.
- 🚧 the default `.simsiorc` is not deployed, so `configs.SimsQuery` carries a
  hardcoded `uuid_regex` fallback for old rc files (`DEL` marked). Shipping one
  would let that go and would fix the drift between per-project rc files
  (e.g. `umps-tests` lacks `uuid_regex`, `template`, `unsafe_update`).
- 💡 `Simulation.link` should reject keys reserved by `[IO-handlers]` and assert the
  resolved path stays under an rc-declared directory (both `TODO` in source).
- 💡 `sim_to_uid`, `uid_to_sim` (like `group_to_path` and vice versa).
- 💡 allow specifying a config path instead of a group name (especially in `run_sim`).
- 💡 `sim_class` configurable from `.simsiorc` (`TODO` in `runsim.run_sim`).
- 💡 `runtime_info(ext_cpu_time=...)` is an ugly hook for `ext_qtea`; find a cleaner
  accounting mechanism.
- ✅ `sims_or_group_arg` returns `Simulation`, not uid.
- ✅ `Simulation == Simulation.uid`, use `is` to distinguish.

## analysis: quantities & filters

- 🩹 `Measure` cache keys hash on `repr` = `"Measure:<name>"`, so two distinct
  measures sharing a name collide. `Measure.__init__` papers over this by purging
  caches on redefinition; the warning about overwriting is currently suppressed
  (`quantities.py`, "output a sensible amount of warnings").
- 🩹 `filters.sqrt_`/`mean_` labels: `nomath` in the label breaks non-math text
  (marked `FIXME` in source).
- 💡 drop the `mplotter` import from `analysis/filters` (marked `DEL`); the analysis
  layer should not hard-depend on a personal plotting fork. Gates a release.
- 💡 `Function == Function.name`, `is` to distinguish? Justified if we make them
  singletons, but we still need to allow copies with different attrs. Then
  `F is not F_copy`, which seems fine because `==` should also have returned False
  if attrs differ.

## analysis: organize & grids

- 🩹 `organize.uids_grid` warns `discarded N simulations with duplicate coords`
  without saying *which* coords collided — the usual cause is an under-determined
  key list, but the message gives nothing to act on. Report the offending coord
  tuples, or at least the count per duplicate group.
- 🚧 **numpy → xarray promotion**: grids are `DataArray`s now, but plotting still
  round-trips through masked arrays (`_prepare_uids_grid` → `to_masked_array`).
  Decide whether `Function`/`Measure` objects should become real xarray coords (see
  the commented-out block in `uids_grid`) — that would let `transpose_grid` drop its
  `.name` juggling and the `HACK` branch.
- ✅ warn on duplicate params in `uids_grid`.

## analysis: plotting

- 🩹 `grid_titles` places row/column titles with `add_axis_label`, which pads from
  the axes edge only: the row title lands on top of the y-axis label (and, with
  unshared axes, of the tick labels). Needs a bbox including
  ticks/ticklabels/label (`TODO` in `add_axis_label`), or `fig.align_labels`-style
  post-placement. A layout engine would *not* fix this on its own: these titles are
  bare `ax.text`, which no engine accounts for. Making column titles real
  `ax.set_title` and row titles the `ylabel` of the last column (label on the right)
  would, and would help under `axg.Grid` too.
- 🚧 the `report_nd` refactor was left half-done (commit `3446645`, "bad partial
  refactor"); the ugrid/xarray axis transposition in `transpose_grid` is the
  fragile part.
- 🚧 `report_1d` accepts a `cycler` grid dimension but it is not implemented
  (`# DEL cycler=None hack while cycler not implemented`).
- 🚧 `grid_from_obs` is a sketch raising `NotImplementedError`; would give
  `report_2d` real (non-index) image extents. Note `imshow` cannot handle
  non-linear coords — needs `pcolormesh` for the general case.
- 💡 **layout backend**: `axg.Grid` gives fixed-size tiles but pads them blindly, so
  whatever reaches into the gaps has to be measured by hand (`_fit_axes_pad`: a
  `get_tightbbox` pass, single, so stale if the caller reformats the axes
  afterwards). The alternative is
  `fig.subplots(layout="constrained")`, which pads from the real drawn extents and
  also places the colorbar, letting `report_1d` drop its manual
  `div.append_size`/`new_locator` block. Costs: tiles are no longer a fixed size
  across figures, and `report_2d` should *not* move — constrained layout handles the
  fixed-aspect `imshow` tiles of `ImageGrid` badly (`layout="compressed"` only
  mitigates it). The version worth doing is not a swap but a thin axes container
  exposing `axes_row`/`axes_column`/`__iter__`/`cax`, with the backend selected by a
  `layout=` kwarg (`"fixed"` → `Grid`/`ImageGrid`, `"constrained"` → `subplots`).
  `grid_titles`, `axes_1d` and `plot_2d_data` already need nothing else, and it is
  the same abstraction `report_nd` wants.
- 💡 use the `DataArray` name in plotting to set the figure path.

## packaging, docs & release

Not urgent, but this is the list that eventually gates a PyPI release.

- 💡 a smoke test suite — the cheapest useful version is a fixture project with two
  tiny configs, exercising `cfg_gen` → `run_sim` → `uids_grid` → `Measure`.
- 💡 README is three lines; the two agent skills in consuming projects are currently
  the only real usage documentation and should be folded back into proper docs.
- 💡 `simsio.analysis` as a namespace package.
- 💡 support python 3.10.
