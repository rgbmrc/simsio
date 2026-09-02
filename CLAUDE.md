# simsio — developer notes

Read this before modifying anything under `src/simsio/`. For *using* simsio from a
project, see the `simsio-run` and `simsio-analyze` skills instead (kept in the
consuming project's `.claude/skills/`); this file is about changing the library.

## Context

The author is a physics postdoc, not a maintainer. simsio is improved **on demand**,
from inside whatever project currently needs it — this repo is normally checked out
as a git submodule (`lib/simsio`) of that project and edited in place. Do not
propose "proper" workflows that require a second clone, a release cycle, or a test
suite that does not exist yet. Do propose the cheapest change that works and leaves
the code in a better state than a pure hack.

The long-term goal is a polished library on PyPI. The short-term constraint is the
author's time. When those conflict, the constraint wins — but record the debt in
`notes/backlog.md`.

## Maturity tiers

Treat these differently:

| module | status | rule |
|---|---|---|
| `settings`, `configs`, `iocore`, `serializers`, `simulations`, `runsim` | **stable core** | in daily use across projects; change conservatively, preserve behaviour, prefer additive fixes |
| `analysis/quantities`, `analysis/filters` | maturing | API mostly settled; refactor freely but keep `Measure`/`Function` semantics |
| `analysis/organize` | alpha | the numpy→xarray promotion is half-done; expect rough edges |
| `analysis/plotting`, `analysis/grids` | experimental | actively reshaped; breaking changes are fine |
| `extensions/*` | per-backend glue | only `ext_tenpy` is currently exercised |

## Conventions

- Terse code. No comments for self-evident lines; a short comment justifying a
  non-obvious approach is welcome and used liberally in this codebase.
- Existing markers are meaningful and worth grepping: `TODO`, `FIXME`, `OPT`
  (optimization opportunity), `DEL` (dead once X lands), `NOTE`, `HACK`.
- `ruff` via pre-commit; per-file `# ruff: noqa:` headers where the style is
  deliberate (lambdas in `quantities`, star-import reminders in `runsim`).
- Commit subjects are gitmoji + scope: `🐛 fix config locking`,
  `analysis/filters: logscale devs`. Keep the scope prefix — it is what makes
  cherry-picking core fixes onto `develop` feasible later.

## Branches

`develop` is the reliable core. Work happens on a feature branch of the moment
(currently `xarray-patches`) which runs well ahead of it. Commits touching the
stable core should be scoped as such so they can be replayed onto `develop`.

## Working in-place from a parent project

The submodule is edited directly and its commits are the only copy of the work.
Therefore:

- **Commit inside the submodule before ending any session that changed it**, even
  as WIP, and bump the gitlink in the parent repo. An unpushed, uncommitted change
  here is one `git submodule update` away from being lost.
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
