# %%
"""Interactive tour of simsio's per-simulation asset registry (branch core/assets).

Writable sims are built directly, not through ``run_sim``: ``Simulation(None, {},
readonly=False)`` mints a uuid and links the rc handles without touching a config,
so there is no config entry to clean up afterwards -- only the output dirs, which
the last cell removes. Nothing here reads or writes an existing simulation.

Runs under this package's own .simsiorc (lib/simsio/.simsiorc), not a host
project's: non-keyed handlers (assets/log/par/res) resolve to flat siblings
`out/$uid.<ext>`, only `aux`/`dat` are per-uid directories.
"""

# %%
# region setup
import shutil
import tempfile
from pathlib import Path

import numpy as np
from simsio import *
from simsio import purge_caches, purge_registry
from simsio.analysis.quantities import Measure
from simsio.serializers import JSONSerializer

made = []  # uids to purge in the last cell


def new_sim():
    sim = Simulation(None, {}, readonly=False)
    made.append(sim.uid)
    return sim


def tree(uid):
    """Everything on disk for a uid: family dirs plus out/'s flat siblings."""
    dirs = {
        d: sorted(p.name for p in Path(f"{d}/{uid}").glob("*"))
        for d in ("out", "data")
        if Path(f"{d}/{uid}").is_dir()
    }
    flat = sorted(p.name for p in Path("out").glob(f"{uid}.*"))
    if flat:
        dirs["out (flat)"] = flat
    return dirs


# endregion

# %%
# region write: stash three assets into two different families
sim = new_sim()
uid = sim.uid

sim["res"]["e0"] = -1.234
sim.stash("dmrg2", {"E": np.arange(4.0)}, via="aux")  # small -> out/
sim.stash("gs", {"fake": "state"}, via="dat")  # bulk  -> data/
sim.stash("exc_k0_n0", {"k": 0.0}, via="dat")
sim.dump()

print(tree(uid))
print(Path(f"out/{uid}.json").read_text())
# endregion

# %%
# region read: reopening links the assets, so no link/serializer/path is needed
purge_registry()  # get_sim memoizes; drop the writable instance first
ro = get_sim(uid)

print("assets   :", list(ro.assets))
print("handles  :", sorted(ro.handles))
print("gs       :", ro["gs"])
print("dmrg2 E  :", ro["dmrg2"]["E"])
ro  # noqa: B018  # html repr: uid + par/log links + collapsed asset list
# endregion

# %%
# region the record pins the encoding, not the location
# `via` names an rc handler and the serializer is resolved and written down, so
# nothing in the file freezes the layout -- which is what makes a moved tree work.
print(ro.assets["gs"])
print("resolved  :", ro.handles["gs"].storage)
# endregion

# %%
# region validation: what link refuses, and why
for key, kw in [
    ("res", {"via": "dat"}),  # reserved: res is a storage
    ("dat", {}),  # dat is a family, not a storage
    ("x", {"via": "res"}),  # res has no $key: cannot host an asset
    ("sub/x", {"via": "dat"}),  # a separator would escape the family dir
    ("chi_0.5", {"via": "dat"}),  # with_suffix would eat ".5"
    ("y", {"via": "aux", "serializer": ""}),  # family declares no serializer
]:
    try:
        ro.link(key, **kw)
    except ValueError as e:  # every guard raises ValueError
        print(f"{key:10} {type(e).__name__}: {e}")
# endregion

# %%
# region a re-run adds to the record, it never prunes it
again = Simulation(uid, {}, readonly=False)  # same uid: restores the registry
again.stash("exc_k1_n0", {"k": 0.1}, via="dat")
again.dump()
print("after re-run:", list(again.assets))  # exc_k0_n0 survived
print(tree(uid))
# endregion

# %%
# region a missing file is loud, never silently dropped
Path(f"data/{uid}/exc_k1_n0.pkl").unlink()
purge_registry()
gone = get_sim(uid)  # logs: Registered assets with no file: exc_k1_n0
print("still registered:", "exc_k1_n0" in gone.assets)
try:
    gone["exc_k1_n0"]
except FileNotFoundError as e:
    print("access ->", type(e).__name__, e.filename)
gone
# endregion

# %%
# region sharing: results/ alone, resolved through someone else's .simsiorc
# assemble a rc#1-style results/$uid/ tree from our flat/split rc#2 layout, no data/
share = Path(tempfile.mkdtemp())
(share / "results" / uid).mkdir(parents=True)
shutil.copy(f"out/{uid}.json", share / "results" / uid / "assets.json")
shutil.copy(f"out/{uid}/dmrg2.npz", share / "results" / uid / "dmrg2.npz")
(share / ".simsiorc").write_text(f"""[IO-handlers]
assets = {share}/results/$uid/assets, w, simsio.serializers.JSONSerializer
aux = {share}/results/$uid/$key, w, simsio.serializers.NPZSerializer
dat = {share}/elsewhere/$uid/$key, w, simsio.serializers.PickleSerializer
log = {share}/results/$uid/info,  , simsio.serializers.LogSerializer
par = {share}/results/$uid/params, w, simsio.extensions.ext_tenpy.TeNPyYAMLSerializer
res = {share}/results/$uid/measures, w, simsio.serializers.NPZSerializer
""")

with rc_context(str(share / ".simsiorc")):
    purge_registry()
    guest = get_sim(uid)
    print("assets known  :", list(guest.assets))
    print("shipped asset :", guest["dmrg2"]["E"])  # travelled inside results/
    try:
        guest["gs"]  # data/ was not shipped
    except FileNotFoundError as e:
        print("absent asset  ->", e.filename)  # the *guest's* dat template
purge_registry()
shutil.rmtree(share)
guest
# endregion

# %%
# region stash is linear: dump() must not rewrite what is already written
s2 = new_sim()
s2.stash("big", {"x": np.zeros(10)}, via="dat")
before = s2.handles["big"].storage.stat().st_mtime_ns
s2["res"]["anything"] = 1
s2.dump()  # rewrites res, not big
after = s2.handles["big"].storage.stat().st_mtime_ns
print("untouched by dump:", before == after, "| still cached:", "big" in s2)
# endregion

# %%
# region a heterogeneous family: the serializer is per asset, and recorded
s3 = new_sim()
s3.stash("as_npz", {"v": np.arange(3)}, via="aux")
s3.stash("as_json", {"nested": {"deep": [1, 2]}}, via="aux", serializer=JSONSerializer)
s3.dump()
print(tree(s3.uid)["out"])
print({k: v["serializer"].rsplit(".", 1)[-1] for k, v in s3.assets.items()})
# dpath sees nested *content*, which is why hierarchy belongs there and not in the
# key namespace: a key holding "/" is unreachable, dpath splits the glob on it.
# Mind the wart (backlog): dpath globs by enumeration and a Cache enumerates only
# what it has loaded, so a path Measure misses until the handle is touched -- and
# the Measure cache then makes that miss stick.
purge_registry()
cold = get_sim(s3.uid)
print("cold  :", Measure.get("as_json/nested/deep")(cold))  # Ellipsis: not loaded yet
cold["as_json"]  # load the handle...
purge_caches()  # ...and drop the cached miss
print("warm  :", Measure.get("as_json/nested/deep")(cold))
print("direct:", cold["as_json"]["nested"]["deep"])
# endregion

# %%
# region cleanup: no config entries were made, only output dirs/files
for u in made:
    for d in ("out", "data"):
        shutil.rmtree(f"{d}/{u}", ignore_errors=True)
    for p in Path("out").glob(f"{u}.*"):
        p.unlink()
purge_registry()
print("purged:", made)
made.clear()
# endregion
