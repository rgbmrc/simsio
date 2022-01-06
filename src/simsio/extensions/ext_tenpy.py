import h5py
from tenpy.tools.params import Config, asConfig
from tenpy.algorithms.truncation import TruncationError
from tenpy.tools.hdf5_io import load_from_hdf5, save_to_hdf5

from simsio.serializers import YAMLSerializer


class TeNPyYAMLSerializer(YAMLSerializer):
    def __init__(self):
        super().__init__()
        self.yaml.representer.add_multi_representer(
            Config,
            lambda dumper, d: dumper.represent_dict(d.as_dict()),
        )
        self.yaml.representer.add_multi_representer(
            TruncationError,
            lambda dumper, d: dumper.represent_dict(vars(d)),
        )

    def load(self, f):
        return asConfig(super().load(f), "Root")
