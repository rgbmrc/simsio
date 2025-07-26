import fcntl
import logging
import re
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import cached_property, wraps
from itertools import chain, product
from pathlib import Path
from shutil import rmtree
from string import Template

import dpath
import ruamel.yaml as yaml

from simsio.settings import rc

# from simsio.analysis.collect import uids_sort # TODO by sort_configs but leads to circular imports

__all__ = [
    "SimsQuery",
    "sims_or_group_arg",
    "cfg_glob",
    "path_to_group",
    "group_to_path",
    "cfg_lock",
    "cfg_update",
    "cfg_update",
    "cfg_gen",
]

logger = logging.getLogger(__name__)

yamlsf = yaml.YAML(typ="safe")
yamlrt = yaml.YAML(typ="rt")
yamlrt.width = 8192


CFG_EXT = ".yaml"


def get_cfg_dir():
    return Path(rc["configs"]["directory"])


# TODO use file cache for _config_path_history
HISTORY_FILE = ".simsio_history"
_config_path_history = deque(maxlen=100)


def cfg_glob(pattern=None, cron=False):
    """
    Returns the paths of config files matching a glob.

    Parameters
    ----------
    group : str, optional
        Glob pattern to match (extension excluded), by default '*
    configs_dir : str, optional
        Directory where to look for config files, by default CONFIGS_DIR
    configs_ext : str, optional
        Extension of config files, by default CONFIGS_EXT

    Returns
    -------
    list[Path]
        Paths of matching config files, ordered chronologically, from the most recently used process
    """
    pattern = (pattern or "**/*") + CFG_EXT
    paths = get_cfg_dir().glob(pattern)
    if cron:
        paths = set(paths)
        paths = chain(
            (p for p in _config_path_history if p in paths),
            (p for p in paths if p not in _config_path_history),
        )
    return paths


def path_to_group(p):
    return str(p.relative_to(get_cfg_dir()).with_suffix(""))


def group_to_path(g):
    return Path(get_cfg_dir(), g).with_suffix(CFG_EXT)


def _merge(dst, src):
    if isinstance(dst, dict) and isinstance(src, dict):
        for k in src:
            if k not in dst:
                dst[k] = src[k]
            else:
                _merge(dst[k], src[k])


def _expand(config, templates):
    if isinstance(config, dict):
        refs = config.pop(rc["configs"]["header_ref"], [])
        if isinstance(refs, str):
            refs = [refs]
        for k in config:
            _expand(config[k], templates)
        for r in reversed(refs):
            _merge(config, templates[r])
    elif isinstance(config, list):
        for v in config:
            _expand(v, templates)


def cfg_load(uid, group=None, expand=True):
    if uid in {rc["configs"]["header_tag"], rc["configs"]["header_ref"]}:
        raise KeyError(f"Key {uid} is reserved")
    for path in cfg_glob(group):
        cfgs = yamlsf.load(path)
        if cfgs and (cfg := cfgs.get(uid)):
            while path in _config_path_history:
                _config_path_history.remove(path)
            _config_path_history.appendleft(path)
            break
    else:
        raise KeyError(f"Simulation {uid} config not found")

    if expand:  # expand config via templates
        _expand(cfg, cfgs.get(rc["configs"]["header_tag"], {}))

    return path, cfg


@contextmanager
def cfg_lock(path):
    with open(path, "r+") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        logger.debug("Locked config %s", path)
        yield f
        fcntl.lockf(f, fcntl.LOCK_UN)  # probably superflous


@contextmanager
def cfg_update(f):
    cfgs = yamlrt.load(f)
    yield cfgs
    f.seek(0)
    yamlrt.dump(cfgs, f)
    f.truncate()


def cfg_update_uid(path, old_uid, new_uid, template=None):
    path = Path(path)
    if template is None:
        template = rc["configs"].getboolean("template")
    ref_uid = new_uid.rstrip("~R")
    map_uid = {old_uid: ref_uid} if template and ref_uid != old_uid else None
    with cfg_lock(path) as f:
        # >1e3 times faster on O(1e3) lines
        if rc["configs"].getboolean("unsafe_update"):
            old_key = re.compile(rf"^{old_uid}(?=:[^\w])")
            # tempfile.SpooledTemporaryFile for large configs? no point
            # because must still fit in memory when loaded as yaml
            cfg = f.readlines()
            for i, l in enumerate(cfg):
                # update uid
                l = old_key.sub(new_uid, l, 1)
                # template refs
                if map_uid:
                    l = Template(l).safe_substitute(map_uid)
                cfg[i] = l
            # copy only once update has been successfully completed
            # (no shutil.copyfile as it voids the lock on f)
            f.seek(0)
            f.writelines(cfg)
            f.truncate()
        else:
            with cfg_update(f) as cfg:
                # update uid
                cfg.insert(list(cfg).index(old_uid), new_uid, cfg.pop(old_uid))
                # template refs
                if map_uid:
                    for p, leaf in dpath.search(cfg, "**", yielded=True):
                        if isinstance(leaf, str) and old_uid in leaf:
                            dpath.set(cfg, p, Template(leaf).safe_substitute(map_uid))


def cfg_sort(glob, keys):
    raise NotImplementedError
    tag = rc["configs"]["header_tag"]
    for p in cfg_glob(glob):
        with cfg_lock(p) as f:
            with cfg_update(f) as cfg:
                header = cfg.pop(tag, None)
                for u in reversed(uids_sort(cfg, keys)):  # noqa: F821
                    cfg.insert(0, u, cfg.pop(u))
                if header:
                    cfg.insert(0, tag, header)


def cfg_pop(*uids, group=None):
    cfgs_paths = defaultdict(set)
    for u in uids:  # TODO: use SimsQuery
        p, _ = cfg_load(u, group, expand=False)
        cfgs_paths[p].add(u)
    for p, us in cfgs_paths.items():
        with cfg_lock(p) as f:
            with cfg_update(f) as cfg:
                for u in us:
                    for h in rc["IO-handlers"].values():
                        glob = h.split(",")[0].strip()
                        glob = Template(glob).substitute(uid=u, key="*")
                        for p in Path(dir).glob(u):  # TODO: make recursive?
                            if p.is_file():
                                p.unlink()
                            elif p.is_dir():
                                rmtree(p)
                    cfg.pop(u)
                    logger.info(f"Deleted {u}")


def cfg_gen(template, params, glob=None):
    generated = {}
    for path in cfg_glob(glob):
        configs = yamlrt.load(path)
        header = configs[rc["configs"]["header_tag"]]
        template = Template(header[template])
        uids = generated[path.stem] = set()
        try:
            keys = params.keys()
            vals = params.values()
        except AttributeError:
            pass
        else:
            params = [dict(zip(keys, vs)) for vs in product(*vals)]
        for prev, ps in enumerate(params):
            yml = template.substitute(ps, enum=prev + 1, prev=prev)
            c = yamlrt.load(yml)
            configs |= c
            uids |= set(c)
        yamlrt.dump(configs, path)
    return generated


def sims_or_group_arg(func_sims):
    @wraps(func_sims)
    def func_sims_or_group(sims_or_group, *args, **kwargs):
        if isinstance(sims_or_group, str):
            sims_or_group = SimsQuery(sims_or_group)
        return func_sims(sims_or_group, *args, **kwargs)

    return func_sims_or_group


class SimsQuery:
    def __init__(self, *group_globs, valid_uuid=True, select=None):
        self.group_globs = group_globs or ["**/*"]
        self.valid_uuid = valid_uuid
        self.select = select

    @cached_property
    def groups(self):
        if self.valid_uuid:
            # hardcoded default for backward compatibility with old .simsiorc files
            # DEL when default rc file is deployed
            uuid_regex = rc["configs"].get("uuid_regex", "[a-z0-9]{32}")
            select = re.compile(uuid_regex, re.S).fullmatch
        else:
            select = rc["configs"]["header_tag"].__ne__
        if self.select is not None:
            select = lambda u: select(u) and self.select(u)  # noqa: E731
        # keeps any config that maches a glob, even if no selected uids
        return {
            path_to_group(p): [*filter(select, yamlsf.load(p) or [])]
            for glob in self.group_globs
            for p in cfg_glob(glob)
        }

    @cached_property
    def uids(self):
        return {u: g for g, us in self.groups.items() for u in us}

    def __iter__(self):
        return iter(self.uids)

    def __len__(self):
        return len(self.uids)

    def __repr__(self):
        args = f"group_globs={self.group_globs}, valid_uuid={self.valid_uuid}"
        return f"{type(self).__name__}({args})"
