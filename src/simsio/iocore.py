from functools import cache
import logging
from collections import UserDict, namedtuple
from importlib import import_module
from pathlib import Path

logger = logging.getLogger(__name__)

IOInfo = namedtuple("IOInfo", ("storage", "write_mode", "serializer"))


def _get_mod_attr(name):
    mod, attr = name.rsplit(".", 1)
    return getattr(import_module(mod), attr)


class IOHandler:
    def __init__(self, readonly=False):
        self.readonly = readonly
        self.handles = {}

    @staticmethod
    def _backup_path(p):
        bak = ".bak"
        try:
            return p.with_suffix(p.suffix + bak)  # p pathlib.Path
        except AttributeError:
            return p + bak  # p string

    def link(self, key, path, write_mode, serializer):

        if isinstance(serializer, str):
            serializer = _get_mod_attr(serializer)()

        path = Path(path).with_suffix(serializer.ext)
        if not self.readonly and write_mode:
            # enusre storge directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            # check storage file write permission
            with open(path, "a"):
                pass

        self.handles[key] = IOInfo(path, write_mode, serializer)
        logger.debug(f"Linked {key} with {path}")

    def load(self, key):
        info = self.handles[key]
        path = info.storage
        with open(path, "r" + info.serializer.typ) as f:
            val = info.serializer.load(f)
        logger.info(f"Loaded {key} from {path}")
        return val

    def dump(self, **keyvals):
        # check permissions
        writable = all(self.handles[k].write_mode for k in keyvals)
        if self.readonly or not writable:
            raise PermissionError("Readonly handles: dump not allowed")

        # backup
        paths = (self.handles[k].storage for k in keyvals)
        backup = [p.rename(self._backup_path(p)) for p in paths if p.is_file()]

        # dump
        for key, val in keyvals.items():
            info = self.handles[key]
            path = info.storage
            with open(path, info.write_mode + info.serializer.typ) as f:
                info.serializer.dump(val, f)
            logger.info(f"Dumped {key} into {path}")

        # cleanup
        for p in backup:
            p.unlink()


class Cache(UserDict, IOHandler):
    def __init__(self, **handler_kw):
        UserDict.__init__(self)
        IOHandler.__init__(self, **handler_kw)

    def __getitem__(self, key):
        if key not in self:
            self.load(key, cache=True)
        return super().__getitem__(key)

    def __setitem__(self, key, val):
        if not key in self.handles:
            raise KeyError(f"Cannot set unlinked IO handle {key}")
        super().__setitem__(key, val)

    def load(self, key, cache=True):
        d = super().load(key)
        if cache:
            self[key] = d
        return d

    @property
    def writable(self):
        return {k: v for k, v in self.items() if self.handles[k].write_mode}
