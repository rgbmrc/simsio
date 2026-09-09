# UCAS (Ufficio Complicazione Affari Semplici)
#
# this could be a O(10) lines file:
#
# from collections import namedtuple
# Serializer = namedtuple('Serializer', ('typ', 'ext', 'load', 'dump'))
# json_serializer = Serializer('t', '.json', json.load, json.dump)
# yaml_serializer = Serializer('t', '.yaml', yaml.load, yaml.dump)
# pkl_serializer = Serializer('b', '.pkl', pickle.load, pickle.dump)
# hkl_serializer = Serializer('b', '.hkl', hickle.load, hickle.dump)
# npy_serializer = Serializer('b', '.npy', np.load, lambda d, f: np.save(f, d))
# npz_serializer = Serializer('b', '.npz', lambda f: dict(np.load(f)), lambda d, f: np.savez(f, **d))
#
# instead: metaclassess LoL ...

import json
import pickle

import numpy as np
import ruamel.yaml as ryaml


def _eof(f):
    pos = f.tell()
    eof = not f.read(1)
    eof or f.seek(pos)
    return eof


class SerializerMeta(type):
    def __new__(typ, *args, **kwds):
        cls = super().__new__(typ, *args, **kwds)
        assert callable(cls.load), '"load" must be callable'
        assert callable(cls.dump), '"dump" must be callable'
        assert cls.typ in ("", "b", "t"), '"typ" must be "", "b" or "t"'
        assert isinstance(cls.ext, str), '"ext" must br a string'
        return cls


# region: bin serializers


class PickleSerializer(metaclass=SerializerMeta):
    typ = "b"
    ext = ".pkl"
    load = staticmethod(pickle.load)
    dump = staticmethod(pickle.dump)


class NPYSerializer(metaclass=SerializerMeta):
    typ = "b"
    ext = ".npy"

    @staticmethod
    def load(f):
        return {} if _eof(f) else np.load(f)

    @staticmethod
    def dump(d, f):
        np.save(f, d)


class NPZSerializer(metaclass=SerializerMeta):
    typ = "b"
    ext = ".npz"

    @staticmethod
    def load(f):
        if _eof(f):
            return {}
        with np.load(f) as data:
            return dict(data)  # np.load is lazy for .npz files

    @staticmethod
    def dump(d, f):
        np.savez(f, **d)


# endregion

# region: txt serializers


class TxtSerializer(metaclass=SerializerMeta):
    typ = "t"
    ext = ".txt"

    @staticmethod
    def load(f):
        return f.read()

    @staticmethod
    def dump(d, f):
        f.write(d)


class LogSerializer(TxtSerializer):
    ext = ".log"


class JSONSerializer(metaclass=SerializerMeta):
    typ = "t"
    ext = ".json"

    @staticmethod
    def load(f):
        return {} if _eof(f) else json.load(f)

    @staticmethod
    def dump(d, f):
        # tolist() covers numpy scalars and arrays, which have no __dict__
        default = lambda o: o.tolist() if hasattr(o, "tolist") else vars(o)
        json.dump(d, f, indent=2, default=default)


class YAMLSerializer(metaclass=SerializerMeta):
    typ = "t"
    ext = ".yaml"

    def _make(self):
        # a YAML instance cannot be pickled/deepcopied
        # if stored as an attribute, xarray.datarray.copy() breaks
        yaml = ryaml.YAML(typ="safe")
        yaml.default_flow_style = False
        yaml.representer.ignore_aliases = lambda *args: True
        yaml.representer.add_multi_representer(
            np.generic,
            lambda dumper, data: dumper.represent_data(data.item()),
        )
        return yaml

    def load(self, f):
        return {} if _eof(f) else self._make().load(f)

    def dump(self, d, f):
        self._make().dump(d, f)


# endregion
