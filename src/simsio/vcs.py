"""Per-launch git snapshot of a worktree and its submodules.

``python -m simsio.vcs [label]`` prints a commit sha reproducing the current worktree,
tracked and untracked (non-ignored) files alike. It is built in a temporary copy of
the index and anchored by ``refs/simsio/<label>-<timestamp>``, so HEAD, branches and
the user's staging area are untouched; a clean tree yields HEAD and no ref. Launch
scripts export it as ``SIMSIO_SNAPSHOT``, which writable simulations record as
``versioning.git_snapshot`` through the rc entry
``[versioning] git_snapshot = printenv SIMSIO_SNAPSHOT``::

    SIMSIO_SNAPSHOT=$(python -m simsio.vcs "$(basename "$0" .sh)") || exit
    export SIMSIO_SNAPSHOT  # separate line: https://www.shellcheck.net/wiki/SC2155

To push snapshot refs::

    git submodule foreach --recursive 'git push origin "refs/simsio/*:refs/simsio/*" || :'
    git push origin 'refs/simsio/*:refs/simsio/*'

Beware: this uploads every untracked file the snapshots captured, each up to
``[versioning.snapshot] max_untracked`` in size, and the remote keeps them for good.

"""

import argparse
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from simsio.settings import rc

__all__ = ["snapshot"]

logger = logging.getLogger(__name__)


def _git(path, *args, env=None, input=None):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        env=env,
        input=input,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _size(s):
    """Parse a size like ``10M`` into bytes."""
    s = s.strip().upper()
    exp = "KMG".find(s[-1]) + 1
    return int(float(s[:-1] if exp else s) * 1024**exp)


def snapshot(path=".", label="", stamp=None, max_untracked=None):
    """Snapshot the worktree at `path` (submodules first), returning a commit sha.

    Untracked files larger than `max_untracked` bytes are skipped with a warning
    (default from ``[versioning.snapshot] max_untracked`` in the rc, else 10M). Returns None
    outside a git repository.
    """
    try:
        top = Path(_git(path, "rev-parse", "--show-toplevel"))
    except subprocess.CalledProcessError:
        logger.warning("%s is not in a git repository: no snapshot", path)
        return None
    stamp = stamp or time.strftime("%Y%m%dT%H%M%S")
    if max_untracked is None:
        max_untracked = _size(
            rc.get("versioning.snapshot", "max_untracked", fallback="10M")
        )
    gitlinks = {
        sub: snapshot(top / sub, label, stamp, max_untracked)
        for line in _git(top, "ls-files", "-s").splitlines()
        if line.startswith("160000")
        and (top / (sub := line.split("\t", 1)[1]) / ".git").exists()
    }
    index = _git(top, "rev-parse", "--path-format=absolute", "--git-path", "index")
    with TemporaryDirectory() as tmp:
        # a copy of the real index keeps its stat cache: only modified files rehash
        env = os.environ | {"GIT_INDEX_FILE": shutil.copy(index, tmp)}
        _git(top, "add", "-u", env=env)
        untracked = _git(top, "ls-files", "-oz", "--exclude-standard", env=env)
        small, large = [], []
        for f in filter(None, untracked.split("\0")):
            (small if (top / f).lstat().st_size <= max_untracked else large).append(f)
        if large:
            logger.warning("Untracked files left out of the snapshot: %s", large)
        if small:
            add = ("add", "--pathspec-from-file=-", "--pathspec-file-nul")
            _git(top, *add, env=env, input="\0".join(small))
        for sub, sha in gitlinks.items():
            _git(top, "update-index", "--cacheinfo", f"160000,{sha},{sub}", env=env)
        tree = _git(top, "write-tree", env=env)
    head = _git(top, "rev-parse", "HEAD")
    if tree == _git(top, "rev-parse", "HEAD^{tree}"):
        return head
    name = "-".join(filter(None, (label, stamp)))
    sha = _git(top, "commit-tree", tree, "-p", head, "-m", f"simsio snapshot {name}")
    _git(top, "update-ref", f"refs/simsio/{name}", sha)
    logger.info("%s: refs/simsio/%s -> %s", top, name, sha)
    return sha


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("label", nargs="?", default="", help="ref name prefix")
    logging.basicConfig(format="simsio.vcs %(levelname)s | %(message)s", level="INFO")
    print(snapshot(label=parser.parse_args().label) or "")


if __name__ == "__main__":
    main()
