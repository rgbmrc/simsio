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
- 🐛 `Simulation.__copy__` shares `handles` with the original (`UserDict.__copy__`
  only updates `__dict__`) while assigning a fresh `uid`, so the copy's storages
  still point at the original's paths and linking on the copy mutates the original.
  Should re-link every handle under the new uid.
- 🐛 `JSONSerializer.dump` passes `default=lambda o: vars(o)`, which raises an opaque
  `TypeError` on numpy scalars. Wants `o.tolist() if hasattr(o, "tolist") else vars(o)`.
- 🩹 `IOHandler.dump` renames each storage to `.bak` before writing, so a concurrent
  reader can see the file missing for a moment (opens as `{}`/`FileNotFoundError`).
  Pre-existing for `par`/`res`; the assets registry inherits it.
- 🩹 `extensions.ext_tenpy.TeNPySimulation.close` reads `self.par`, which does not
  exist (it is `self["par"]`); the `AttributeError` is swallowed, so `warn_unused`
  has never fired from there. Fixing the typo needs
  `par.touch("uuid", "versioning", "monitoring")` in the same commit, or it warns
  about simsio's own bookkeeping keys on every run.
- 🚧 `configs._config_path_history` should be backed by `HISTORY_FILE`
  (`.simsio_history`), currently an in-process deque only.
- 🚧 the default `.simsiorc` is not deployed, so `configs.SimsQuery` carries a
  hardcoded `uuid_regex` fallback for old rc files (`DEL` marked). Shipping one
  would let that go and would fix the drift between per-project rc files
  (e.g. `umps-tests` lacks `uuid_regex`, `template`, `unsafe_update`).
- 💡 `sim_to_uid`, `uid_to_sim` (like `group_to_path` and vice versa).
- 💡 allow specifying a config path instead of a group name (especially in `run_sim`).
- 💡 `sim_class` configurable from `.simsiorc` (`TODO` in `runsim.run_sim`).
- 💡 `runtime_info(ext_cpu_time=...)` is an ugly hook for `ext_qtea`; find a cleaner
  accounting mechanism.
- 💡 deprecate the `path=` override in `Simulation.link`: it is the only remaining
  way to place a storage outside the rc layout (a template-resolved path cannot
  escape, the key being a single component, and `_register` rejects `path=` — so the
  hole is readonly-only and writes nothing). Dropping it retires the last of the
  "assert path stays under an rc directory" TODO.
- 💡 `cfg_pop` should purge an asset's storages via the registry once it is fixed;
  the project-side `sims.sh` helpers likewise only scan `results/`, so `data/`
  assets leak from `purge`/`quota`.
- 💡 `_repr_html_`'s "dynamic keys" TODO is now answerable: link `self.assets` too.
- ✅ `sims_or_group_arg` returns `Simulation`, not uid.
- ✅ `Simulation == Simulation.uid`, use `is` to distinguish.
- ✅ `Simulation.link` rejects keys reserved by `[IO-handlers]` and asserts the
  resolved path stays under an rc-declared directory (done with the asset registry;
  a template-resolved path can no longer escape either, the key being a single path
  component).

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

- 🐛 `report_1d(..., y_titles=None)` raises `x and y must have same first dimension`
  from `plot_1d_data` when `x_obs` is *stacked*, as `filters.dev_obs` returns it:
  dropping the row dimension leaves `_prepare_arrays` broadcasting `x_obs` against a
  grid axis that `y_obs` does not have. A plain `x_obs` is fine, and the dim
  resolution (`transpose_grid`) is not the culprit — it resolves the same slots with
  or without the crash. `analysis/verify_model.py:194` calls it that way.
- 🩹 the figure size ignores whatever the grid draws *outside* its tiles — row and
  column titles, the edge axis labels, the colorbar — so those clip in a plain
  `savefig` (inline backends and `bbox_inches="tight"` hide the problem).
  `AxesGrid.fit_axes_pad` only reports the width it adds to the colorbar pad, which
  `_report_titles` gives back to the figure. Related: `TILE_SIZE` is not the size a
  tile gets, since the `SubplotDivider` lays the grid out inside the fractional
  subplot margins. Both go away by measuring the outer overhangs (as `grid_titles`
  already does per side), sizing the figure as `margins + tiles + pads (+ cbar)` and
  pinning the divider rect to those margins in figure fractions.
- 🩹 `annotate_image_axis` labels *every* coordinate value, so a `report_2d` grid of
  narrow tiles overlaps its own x tick labels (`$1e-08$$1e-07$...`). It should thin
  them out, or leave a real Locator/Formatter in place of the FixedLocator.
- 🩹 `AxesGrid._init_locators` is ported from mpl 3.10's `ImageGrid._init_locators`
  and reads its private `_colorbar_*` state, so a matplotlib upgrade can break it.
  Two deliberate divergences: tiles are `Size.Scaled(1)` unless `aspect` is on, and a
  relative `cbar_size` ("5%") in `cbar_mode="single"` refers to the grid extent along
  the bar rather than across it, as mpl has it.
- 🩹 `title_sides` onto the side a per-tile colorbar sits on (`cbar_mode="each"`)
  grows *every* bar's pad, though only the titled edge row/column has a title to
  clear: `_cbar_pad_size` is one shared `Size.Fixed`, and per-slot pads would make
  the tiles unequal. Harmless but airy; the other cbar modes are exact.
- 💡 grid titles are bare `ax.text` placed by `add_axis_label`, past the decorations
  measured on that side. That is deliberate — it is uniform across the four sides,
  and it keeps the titles of a row/column whose tile was switched off
  (`ax.axis("off")`) — but it is a one-shot measurement, and no layout engine
  accounts for a bare `Text`. Under a constrained-layout backend they would have to
  become real decorators: `ax.set_title` for the column titles, and for the row
  titles the `ylabel` of the last column (`label_position="right"`) or a
  label-only `secondary_yaxis`. Note both then need a *free* side, which the
  colorbar of `report_1d`/`report_2d` may well be occupying.
- 🚧 the `report_nd` refactor was left half-done (commit `3446645`, "bad partial
  refactor"). Its idea — the report args transpose the ugrid onto the report slots —
  is kept and now resolves named dims before positional ones; `report_1d` and
  `report_2d` share the whole skeleton, so what is left is to fold them into one
  entry point taking the slot list (`REPORT_1D_DIMS` / `REPORT_2D_DIMS`) and the
  inner loop. Note `x_obs`/`y_obs` are still *not* slots in `report_2d`: we do not
  know what `obs` does with the dims past the grid.
- 🚧 `report_1d` accepts a `cycler` grid dimension but it is not implemented
  (`# DEL cycler=None hack while cycler not implemented`).
- 🚧 `grid_from_obs` is a sketch raising `NotImplementedError`; would give
  `report_2d` real (non-index) image extents. Note `imshow` cannot handle
  non-linear coords — needs `pcolormesh` for the general case.
- 💡 **layout backend**: `AxesGrid` gives fixed-size tiles but pads them blindly, so
  whatever reaches into the gaps is measured by hand (`fit_axes_pad`: one
  `get_tightbbox` pass, so it goes stale if the caller reformats the axes
  afterwards). The alternative is `fig.subplots(layout="constrained")`, which pads
  from the real drawn extents and also places the colorbar. Costs: tiles are no
  longer a fixed size across figures, and `report_2d` should *not* move —
  constrained layout handles fixed-aspect `imshow` tiles badly (`layout="compressed"`
  only mitigates it). Now that `AxesGrid` *is* the axes container the reports talk
  to, this reduces to a second implementation of its surface
  (`axes_row`/`axes_column`/`__iter__`/`cbar_axes`/`ax.cax`/`set_label_mode`/
  `needs_pad_fit`/`fit_axes_pad`), chosen by a `layout=` kwarg.
- 💡 use the `DataArray` name in plotting to set the figure path.

## packaging, docs & release

Not urgent, but this is the list that eventually gates a PyPI release.

- 💡 a smoke test suite — the cheapest useful version is a fixture project with two
  tiny configs, exercising `cfg_gen` → `run_sim` → `uids_grid` → `Measure`.
- 💡 README is three lines; the two agent skills in consuming projects are currently
  the only real usage documentation and should be folded back into proper docs.
- 💡 `simsio.analysis` as a namespace package.
- 💡 support python 3.10.
