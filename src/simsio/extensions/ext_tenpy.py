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


class TeNPySimulation(Simulation):
    """`par` is a tenpy `Config` only while writing: unused-option tracking belongs to a
    running simulation, and would otherwise warn whenever a read-only one is collected."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.readonly:  # after the cfg merge, which must run on plain data
            self["par"] = asConfig(self["par"], "Root")

    def close(self):
        if not self.readonly:
            self["par"].warn_unused(recursive=True)
        super().close()
