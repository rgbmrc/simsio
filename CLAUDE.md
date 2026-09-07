# simsio — developer notes

Read this before modifying anything under `src/simsio/`. For *using* simsio from a
project, see the `simsio-run` and `simsio-analyze` skills instead (kept in the
consuming project's `.claude/skills/`); this file is about changing the library.

## Context

The author is a physics postdoc, not a software maintainer. simsio is improved **on demand**,
from inside whatever project currently needs it — this repo is normally checked out
as a git submodule (`lib/simsio`) of that project and edited in place. Do not
propose workflows that require a second clone, a release cycle, or a test
suite that does not exist yet.

The long-term goal is a polished library on PyPI. The short-term constraint is the
author's time. When those conflict, the constraint wins — but record the debt in
`notes/backlog.md`.

## Maturity tiers

Treat these differently:

| module | status | rule |
|---|---|---|
| `settings`, `configs`, `iocore`, `serializers`, `simulations`, `runsim` | **stable core** | in daily use across projects; change conservatively |
| `analysis/quantities`, `analysis/filters` | maturing | API mostly settled; refactor freely but keep `Measure`/`Function` semantics |
| `analysis/organize` | alpha | the numpy→xarray promotion is half-done; expect rough edges |
| `analysis/plotting`, `analysis/axes_grid`, `analysis/grids` | experimental | actively reshaped; breaking changes are fine |
| `extensions/*` | per-backend glue | only `ext_tenpy` is currently exercised |

## Conventions

- Terse code. No comments for self-evident lines; a short comment justifying a
  non-obvious approach is welcome.
- Existing markers are meaningful and worth grepping: `TODO`, `FIXME`, `OPT`
  (optimization opportunity), `DEL` (dead once X lands), `NOTE`, `HACK`.
- `ruff` via pre-commit; per-file `# ruff: noqa:` headers where the style is
  deliberate (e.g. lambdas in `quantities`).
- Commit messages are "gitmoji + scope: short description", e.g. `🐛 configs: fix
  locking`, `✨ analysis.filters: logscale devs`.
- Standard emojis: ✨ feature, 🐛 bug, 🚑 critical hotfix, 🩹 non-critical fix,
  ♻️ refactor, 💥 breaking change, 🚧 WIP, 💩 bad code needing rework, 🔧 config,
  ⬆️/⬇️ deps, ✏️ typo, 🎨 format/structure, ⚡️ performance, 🚨 linter, 📝 docs,
  💬 text/literals, 💡 comments/ideas, 🦺 validation, 🙂 UX,
  🍻 hacky code, 🤖 agents files (CLAUDE.md, skills).
- Classify more severely in the stable core (wrong results or data loss are
  🚑, a crash on a supported path is 🐛), less so for analysis and plotting
  (mostly patches 🩹 of experimental code).

## Asset registry invariants

`Simulation` links the `[IO-handlers]` entries whose template has no `$key`; a `$key`
template (`dat`, `aux`) declares an *asset family*, resolved on demand by
`link(key, via=...)`. Extra keys linked on a writable simulation are recorded in the
`assets` handle and re-linked (never loaded) when the simulation is reopened. Four
invariants, each load-bearing — do not "simplify" them away:

- **`.simsiorc` owns location, the registry owns encoding.** The record stores the
  *handler name* plus the serializer actually used, never a resolved path, so a shared
  or relocated tree resolves through the receiver's rc. `_register` rejects `path=`
  for this reason.
- **It is written only through `dump`** (or `stash`, which dumps one asset plus the
  registry). No write-on-link, so a crash leaves no registration without a file.
- **No locking**, because `from_config` already renames the config key to `<uuid>~R`
  under an `fcntl` lock — one writer per uid — and readonly opens never write.
- **Union, never prune.** A re-run adds to the record; missing files are warned about
  once at open. Nothing is deleted implicitly (`unlink` is the explicit way).

An rc without an `assets` entry disables all of it, which is the backward-compatibility
story: no migration for simulations written before this existed.

## Branches

`develop` is the base branch. A one-commit bugfix or patch can go straight onto
it. Anything else starts a new feature branch (small changes can always be
fast-forwarded into `develop` if needed).

`main` is an occasional integration snapshot, behind `develop`: it holds a
stable version of simsio's core library and a preliminary, pre-xarray draft of
the analysis subpackage.

## Working in-place from a parent project

The submodule is edited directly and its commits are the only copy of the work.
Therefore:

- **Commit inside the submodule before ending any session that changed it**, even
  as WIP: an uncommitted change here is one `git submodule update` away from being
  lost. Bumping the gitlink in the parent repo is a separate, optional step — the
  parent's history does not need to track every simsio commit, so do it when it is
  convenient rather than by reflex.
- Never run `git submodule update` in the parent without checking
  `git -C lib/simsio status` first.
- Test against the parent project's real data — that is the point of this setup.
  There is no test suite; a scratch script that exercises the changed path on actual
  simulations is the substitute.

## Persistent backlog

`notes/backlog.md` is the memory of unfinished business: known bugs, half-implemented
features, and ideas raised but not built. Append to it whenever something is found
and not fixed. It is committed on purpose — it must survive across sessions and
across projects.

It is grouped by area, not by severity: put an item in the section it belongs to and
mark it with the same gitmoji the commit closing it would carry (🐛 / 🩹 / 🚧 / 💡,
legend at the top of the file). Add a section rather than letting a catch-all grow.
