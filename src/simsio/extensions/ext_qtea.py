import os
import logging
from collections import ChainMap, defaultdict
from inspect import signature
from itertools import chain
from pathlib import Path

import dpath
import numpy as np
import qtealeaves as qtea
from qtealeaves import map_selector
from qtealeaves.observables import TNObservables
from qtealeaves.convergence_parameters import TNConvergenceParameters

from simsio.simulations import Simulation

logger = logging.getLogger(__name__)

try:
    QTEASimulation = qtea.QuantumGreenTeaSimulation
except AttributeError:
    QTEASimulation = qtea.ATTNSimulation  # legacy version


def update_params_with_defaults(func, kwargs):
    ba = signature(func).bind_partial(**kwargs)
    kwargs |= ba.arguments


def gen_seed(seed=None):
    rng = np.random.default_rng(seed)
    seed = rng.integers(1, 4095, 4)
    seed[-1] += not (seed[-1] % 2)
    # tolist avoids "Unknown type <class 'numpy.int64'>"
    return seed.tolist()


def extract_sweep_time_energy(uid):
    try:
        uid = uid.strip("-R")
        # HACK: retreive path
        cnv_file = f"data/{uid}/output/convergence.log"
        # if empty data unpacking fails with ValueError
        t, e = np.loadtxt(cnv_file, skiprows=1, ndmin=2).T
        return t, e
    except (FileNotFoundError, ValueError):
        return np.zeros((2, 0))


def unravel(obs1d, lvals, *, ndim=0, map_type="HilbertCurveMap", argmap=None):
    if argmap is None:
        posmap = map_selector(len(lvals), lvals, map_type)
        argmap = np.lexsort(tuple(zip(*map(reversed, posmap))))
    if not ndim > 0:
        ndim = np.ndim(obs1d) + ndim
    if ndim == 0:  # scalar
        return obs1d
    obs1d = np.asanyarray(obs1d)
    shape = obs1d.shape[:-ndim]
    inds = np.ix_(*map(range, shape), *((argmap,) * ndim))
    return obs1d[inds].reshape(shape + tuple(lvals) * ndim)


class QuantumGreenTeaSimulation(Simulation):

    unravel_classes = {"TNObsLocal", "TNObsCorr"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # shortcuts, so that if .par changes it is only here
        self._p_model = self.par.setdefault("model", {})
        self._p_convergence = self.par.setdefault("convergence", {})
        self._p_measures = self.par.setdefault("measures", [])
        self._p_qtea_sim = self.par.setdefault("qtea_sim", {})
        self._p_qtea_run = self.par.setdefault("qtea_run", {})

    def _init_convergence(self):
        # TODO: TNConvergenceParametersFiniteT
        update_params_with_defaults(TNConvergenceParameters, self._p_convergence)
        return TNConvergenceParameters(**self._p_convergence)

    def _init_observables(self):
        # TODO: args: num_trajectories, do_write_hdf5
        observables = TNObservables()
        # each list entry is a dictionary with a single entry:
        # observable_class: list_of_arguments or None
        for obs_class, obs_args in self._p_measures:
            if obs_class == "TNState2File":
                # HACK: use serializer, retreive path
                obs_args[0] = f"data/{self.uid}/output/{obs_args[0]}"
            obs_class = getattr(qtea.observables, obs_class)
            observables += obs_class(*obs_args)
        return observables

    def _unravel_observable(self, obs, ndim=0):
        return unravel(obs, ndim=ndim, **self._unravel_args)

    def init_qtea_simulation(self, model, operators):
        update_params_with_defaults(QTEASimulation, self._p_qtea_sim)
        self.qtea_sim = QTEASimulation(
            model=model,
            operators=operators,
            convergence=self._init_convergence(),
            observables=self._init_observables(),
            # HACK: use rc settings
            folder_name_input=f"data/{self.uid}/input/",
            folder_name_output=f"data/{self.uid}/output/",
            has_log_file=False,  # logging handled by simsio
            **self._p_qtea_sim,
        )

        # TODO: support parameterized lvals
        posmap = map_selector(model.dim, model.lvals, model.map_type)
        argmap = np.lexsort(tuple(zip(*map(reversed, posmap))))
        self._unravel_args = dict(argmap=argmap, lvals=model.lvals)

        # HACK: TODO: test
        qtea_cnv_log = Path(self.qtea_sim.folder_name_output, "convergence.log")
        # FIXME: handle this elsewhere
        qtea_cnv_log.parent.mkdir(parents=True, exist_ok=True)
        qtea_cnv_log.touch()
        simsio_cnv_log = self.handles["cnv"].storage
        simsio_cnv_log.unlink(missing_ok=True)
        try:
            simsio_cnv_log.hardlink_to(qtea_cnv_log)
        except OSError:
            logger.error("Could not hardlink convergence file.", exc_info=True)

    def run_qtea_simulation(self, overwrite=True):
        # FIXME: allow selection of what to include based on rc
        # for g in globs:
        #     dpath.get(self, g)
        seed = gen_seed(self._p_qtea_run.setdefault("seed", 0))
        self._p_qtea_run |= {
            # TODO: str() cause QTEA doesn't do well with pathlib.Path
            "log_file": str(self.handles["log"].storage.absolute()),
        }
        run_params = self._p_qtea_run | {"seed": seed, **self._p_model}
        # we always run a single thread
        self.qtea_sim.run(run_params, delete_existing_folder=overwrite)

    def _parse_results(self):
        try:
            _, cpu_time = next(
                self.qtea_sim.observables.read_cpu_time_from_log("/", self._p_qtea_run)
            )
        except StopIteration:
            # QTEA's python side does not log CPU time (only simulation time)
            # better so, cause already accounted for by simsio
            logger.warning("Could not read CPU time from log.")
        else:
            self.runtime_info(ext_cpu_time=cpu_time)
        # concatenate static & quenches
        measures_list = chain(
            [self.qtea_sim.get_static_obs(self._p_qtea_run)],
            *self.qtea_sim.get_dynamic_obs(self._p_qtea_run),
        )
        measures = defaultdict(list)
        for step, m in enumerate(measures_list):
            # parse measurements at each step
            m.setdefault("time", 0.0)
            # QTEA returns projective_measurements even when not requested
            proj_meas = m.pop("projective_measurements")
            assert not proj_meas, "projective measurements unsupported"
            # store bipartition entropies in a numpy compatible format
            entropy = (  # QTEA used multiple names over time
                m.pop("bondentropy", None)
                or m.pop("bond_entropy", None)
                or m.pop("bond_entropy0", None)
            )
            if entropy:
                m["entropy_cut"] = list(entropy.keys())
                m["entropy_val"] = list(entropy.values())
            # append to measurments dict
            for k, v in m.items():
                measures[k].append(v)
        # unravel observables of classes listed in self.unravel_classes
        for obs_class, obs_args in self._p_measures:
            if obs_class in self.unravel_classes:
                k = obs_args[0]
                measures[k] = self._unravel_observable(measures[k], ndim=-1)
        # drop trivial time index for statics
        if step == 0:
            for k, vs in measures:
                measures[k] = vs[0]
        self.res |= measures

    def dump(self, wait=0, **keyvals):
        if not self.res:
            self._parse_results()
        super().dump(wait, **keyvals)
