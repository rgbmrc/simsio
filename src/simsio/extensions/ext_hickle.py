import hickle

from simsio.serializers import SerializerMeta


class HickleSerializer(metaclass=SerializerMeta):
    typ = ""  # irrelevant but 'b' gives problems
    # https://github.com/telegraphic/hickle/issues/123
    # fixed in hickle >= 3.4.7
    ext = ".hkl"
    load = staticmethod(hickle.load)
    dump = staticmethod(hickle.dump)
