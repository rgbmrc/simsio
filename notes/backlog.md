# Backlog

Persistent memory of unfinished business. Append freely; move items to a changelog
once the library is mature enough for one. Absorbs the former `ideas.md`.

## Known bugs / rough edges

- `configs.cfg_pop` is broken: iterates `Path(dir).glob(u)` where `dir` is the
  builtin, and the glob ignores the handler path template it just built.
- `configs.cfg_sort` raises `NotImplementedError` on entry; the body references
  `uids_sort`, which cannot be imported without a circular import
  (`configs` ← `analysis.organize`).
- `organize.uids_grid` warns `discarded N simulations with duplicate coords`
  without saying *which* coords collided — the usual cause is an under-determined
  key list, but the message gives nothing to act on. Report the offending
  coord tuples, or at least the count per duplicate group.
- `analysis/plotting.report_nd` refactor was left half-done
  (commit `3446645`, "bad partial refactor"); the ugrid/xarray axis transposition
  in `transpose_grid` is the fragile part.
- `Measure` cache keys hash on `repr` = `"Measure:<name>"`, so two distinct
  measures sharing a name collide. `Measure.__init__` papers over this by purging
  caches on redefinition; the warning about overwriting is currently suppressed
  (`quantities.py`, "output a sensible amount of warnings").
- `filters.sqrt_`/`mean_` labels: `nomath` in the label breaks non-math text
  (marked FIXME in source).

## Half-implemented

- **numpy → xarray promotion** (`analysis/organize`): grids are `DataArray`s now,
  but plotting still round-trips through masked arrays (`_prepare_uids_grid` →
  `to_masked_array`). Decide whether `Function`/`Measure` objects should become
  real xarray coords (see the commented-out block in `uids_grid`) — that would let
  `transpose_grid` drop its `.name` juggling and the `HACK` branch.
- `report_1d` accepts a `cycler` grid dimension but it is not implemented
  (`# DEL cycler=None hack while cycler not implemented`).
- `plotting.grid_from_obs` is a sketch raising `NotImplementedError`; would give
  `report_2d` real (non-index) image extents. Note `imshow` cannot handle
  non-linear coords — needs `pcolormesh` for the general case.
- Default `.simsiorc` is not deployed, so `configs.SimsQuery` carries a hardcoded
  `uuid_regex` fallback for old rc files (`DEL` marked). Shipping a default rc
  would let that go and would fix the drift between per-project rc files
  (e.g. `umps-tests` lacks `uuid_regex`, `template`, `unsafe_update`).
- `configs._config_path_history` should be backed by `HISTORY_FILE`
  (`.simsio_history`), currently an in-process deque only.

## Ideas

- [x] `sims_or_group_arg` returns `Simulation`, not uid
- [x] warn on duplicate params in `uids_grid`
- [x] `Simulation == Simulation.uid`, use `is` to distinguish
- [ ] similarly `Function == Function.name`? `is` to distinguish is justified if we
      make them singletons, but we still need to allow copies with different attrs.
      Then `F is not F_copy`, but that seems fine because `==` should also have
      returned False if attrs differ.
- [ ] `sim_to_uid`, `uid_to_sim` (like `group_to_path` and vice versa)
- [ ] allow specifying a config path instead of a group name (especially in `run_sim`)
- [ ] use the `DataArray` name in plotting to set the figure path
- [ ] `simsio.analysis` as a namespace package
- [ ] support python 3.10
- [ ] `sim_class` configurable from `.simsiorc` (TODO in `runsim.run_sim`)
- [ ] `Simulation.link` should reject keys reserved by `[IO-handlers]` and assert the
      resolved path stays under an rc-declared directory (both TODO in source)
- [ ] `runtime_info(ext_cpu_time=...)` is an ugly hook for `ext_qtea`; find a
      cleaner accounting mechanism

## Toward a publishable library

Not urgent, but this is the list that eventually gates a PyPI release:

- a smoke test suite — the cheapest useful version is a fixture project with two
  tiny configs, exercising `cfg_gen` → `run_sim` → `uids_grid` → `Measure`
- drop the `mplotter` import from `analysis/filters` (marked `DEL`); the analysis
  layer should not hard-depend on a personal plotting fork
- README is three lines; the two agent skills in consuming projects are currently
  the only real usage documentation and should be folded back into proper docs
