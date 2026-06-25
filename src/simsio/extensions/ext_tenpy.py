# ruff: noqa: F401 # TODO imports as reminders

from tenpy.algorithms.truncation import TruncationError
from tenpy.tools.hdf5_io import load_from_hdf5, save_to_hdf5
from tenpy.tools.params import Config, asConfig

from simsio.serializers import YAMLSerializer
from simsio.simulations import Simulation


class TeNPyYAMLSerializer(YAMLSerializer):
    def _make(self):
        yaml = super()._make()
        yaml.representer.add_multi_representer(
            Config,
            lambda dumper, d: dumper.represent_dict(d.as_dict()),
        )
        yaml.representer.add_multi_representer(
            TruncationError,
            lambda dumper, d: dumper.represent_dict(vars(d)),
        )
        return yaml

    def load(self, f):
        return asConfig(super().load(f), "Root")


class TeNPySimulation(Simulation):
    def close(self):
        try:
            self.par.warn_unused(recursive=True)
        except AttributeError:
            pass
        super().close()
