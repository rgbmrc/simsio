# Config claim race: concurrent starts of one entry all ran

2026-10-28. Report on a bug in `Simulation.from_config`: the rename of a config entry
to `<uuid>~R` is meant to let exactly one process run that entry. From 2024-04-01 to
this fix it did not. Every process that started the same entry at about the same time
ran it, each under its own UUID.

**Answer:** From `7558b42` on, the default rename path (`unsafe_update = true`) replaced
the key by regex and did nothing when the key was already gone. The process that lost
the race therefore ran too. Its UUID never entered the config, so its results sat in
`results/` without being listed in any config. In a test with $4$ processes starting the
same $30$ entries in step, every entry ran $4$ times. The fix loads and renames the entry
under one exclusive lock before anything is written, and raises `KeyError` when the key
is missing. With it, each entry runs once, on both rename paths.

## How the claim is meant to work

- `run_sim` starts `from_config(name, group)`. The config key (`n5`) is renamed to
  `<uuid>~R` while the sim runs and to `<uuid>` on `close`.
- With several launchers working through the same list of names (separate machines
  sharing the config over NFS), the rename is the claim. A second launcher no longer
  finds `n5` and skips it.
- The skip is what `run_sim(not_found_ok=True)` was for. A `KeyError` while loading
  was logged as one line and the process exited with code $0$. A claimed entry was then
  not reported as a failure by the launcher (no traceback in `nohup.out`, no xargs exit
  code $123$).
- For the claim to exclude a second run, "the key is still there" and "rename it" must
  happen under one lock.

## History

| commit | date | claim |
|---|---|---|
| `de9dc42` "💬 clearer exceptions in loading simulation" | 2024-03-31 | safe |
| `7558b42` "⚡️ new usafe config update option" | 2024-04-01 | broken |
| `1f7fade` "✨ allow to instatiate config-less simulations" | 2024-08-20 | broken |
| `20434e4` "🐛 fix config locking" | 2026-07-05 | broken |
| this fix | 2026-10-28 | safe |

- **`de9dc42`.** The constructor read the config without a lock, then renamed under an
  exclusive lock with `cfgs.pop(uid)`. Two processes could both read `n5`, but only
  one could pop it. The other got `KeyError`, which `not_found_ok` turned into a skip.
  Load and rename were two steps, but the `pop` made the second one fail. That is
  enough for one winner.
- **`7558b42`** added `unsafe_update` (default `true` in every `.simsiorc` since). It
  rewrites the file line by line, replacing the key by a regex, which is $>10^3$ times
  faster than a yaml round trip on large configs.
  - When the key is absent the substitution matches nothing. The function returns
    without error, the loser keeps running under its own UUID, and its `close` is the
    same silent no-op.
  - The yaml path (`unsafe_update = false`) did `list(cfg).index(uid)`, which raises
    `ValueError`, not `KeyError`. There a lost race became a traceback, not a skip.
- **`1f7fade`** moved the config handling into `from_config` and changed the order.
  - Before, the constructor renamed right after loading, so the window between the
    two was a few lines of code.
  - From here on it is: load, build the `Simulation` (writes `results/<uuid>`, runs the
    `[vcs]` commands), rename. The window is now tens of milliseconds, which is why
    every race in the test below is lost.
- **`20434e4`** fixed something else:
  - readers take a shared lock, so nobody loads a half-written file;
  - `cfg_gen` holds one exclusive lock from read to write.

  The read in `cfg_load` and the rename in `cfg_update_uid` stayed in two separate lock
  sections, so the claim stayed broken.

A second effect of the same window concerns `$name` refs (`groundstate.load: $n1`).
`x1` could load its entry while it still read `$n1`. If `n1` was renamed before `x1`
stored its par, `x1` kept the literal `$n1` in its par.

## Fix

In `simsio.configs`:
- `cfg_claim(uid, new_uid, group, template)` locates the entry, then under one
  exclusive lock re-reads the file, raises `KeyError` if `uid` is gone, expands the
  entry, and renames it (and its `$uid` refs).
- The rename itself is `_rename_uid(f, ...)`, shared by `cfg_claim` and
  `cfg_update_uid`. Both paths now raise `KeyError` when the key is missing, without
  writing:
  - the regex path counts substitutions (`subn`);
  - the yaml path checks the key before dumping.

In `Simulation`:
- `from_config` draws the UUID first (the key itself if it already is one), claims the
  entry, and only then builds the `Simulation` with that UUID and the claimed config.
  The writable constructor accepts a UUID plus a `cfg` since the previous commit, which
  makes this order possible.
- A lost race now raises before anything is written: no second run, no stray
  `results/` directory.
- The config the winner loads is read under the same lock as its rename, so `$name`
  refs are as current as the file.
- `close` logs a warning instead of raising when the `~R` entry has been edited away
  meanwhile. Otherwise a sim that finished would fail at the last step.

In `run_sim`:
- `not_found_ok` is removed. Every loading error (uid not found, entry claimed by
  another run, invalid uid, missing par) raises and exits non-zero.
- The quiet exit with code $0$ was for several independent launchers sharing one config.
  With one launcher per batch (slurm array, one xargs), a missing uid is a real error.
- Under slurm, exit code $0$ also marked a task that never started as COMPLETED. An
  `afterok` dependency then released the tasks waiting on it.

## Check

$4$ processes, each calling `from_config` on the same $30$ entries, with a barrier
before each entry so that all $4$ start it together. Script:
`llm-chats/data/simsio-config-locking/race.py` (scratch, run from the repo root).

| | runs | entries run more than once | runs not listed in the config | what the losers got |
|---|---|---|---|---|
| before (`unsafe_update = true`) | $120$ | $30$ | $90$ | nothing (all ran) |
| after, `unsafe_update = true` | $30$ | $0$ | $0$ | `KeyError` |
| after, `unsafe_update = false` | $30$ | $0$ | $0$ | `KeyError` |

Also checked after the fix:
- `$name` refs resolve through `jobs/cluster.py params`;
- a rerun of a UUID entry with `-g` renames to `~R` and back;
- a missing name and a UUID without a par both exit with code $1$;
- readonly loads of all `statics/**` sims are unchanged.

Not done: the leftover runs of double launches between 2024-04 and now were not
searched for. They would be `results/<uuid>` directories listed in no config, with a
config entry of the same parameters under another UUID.
