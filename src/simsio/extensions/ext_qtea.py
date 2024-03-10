import os
from pathlib import Path
from collections import ChainMap
from inspect import signature

import dpath
import numpy as np
import qtealeaves.observables
from qtealeaves import ATTNSimulation, map_selector
from qtealeaves.observables import TNObservables
from qtealeaves.convergence_parameters import TNConvergenceParameters

from simsio.simulations import Simulation


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


def unravel(obs1d, lvals, *, ndim=None, map_type="HilbertCurveMap", argmap=None):
    if argmap is None:
        posmap = map_selector(len(lvals), lvals, map_type)
        argmap = np.lexsort(tuple(zip(*map(reversed, posmap))))
    ndim = ndim or np.ndim(obs1d)
    if ndim == 0:  # scalar
        return obs1d
    obs1d = np.asanyarray(obs1d)
    shape = obs1d.shape[:-ndim]
    inds = np.ix_(*map(range, shape), *((argmap,) * ndim))
    return obs1d[inds].reshape(shape + tuple(lvals) * ndim)


class QuantumGreenTeaSimulation(Simulation):

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
            obs_class = getattr(qtealeaves.observables, obs_class)
            observables += obs_class(*obs_args)
        return observables

    def _unravel_observable(self, obs):
        return unravel(obs, **self._unravel_args)

    def init_qtea_simulation(self, model, operators):
        update_params_with_defaults(ATTNSimulation, self._p_qtea_sim)
        self.qtea_sim = ATTNSimulation(
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
        simsio_cnv_log.hardlink_to(qtea_cnv_log)

    def run_qtea_simulation(self):
        seed = gen_seed(self._p_qtea_run.setdefault("seed", 0))
        # FIXME: allow selection of what to include based on rc
        # for g in globs:
        #     dpath.get(self, g)
        run_params = self._p_qtea_run | {
            "seed": seed,
            "log_file": self.handles["log"].storage,
        }
        # we always run a single thread
        self.qtea_sim.run(run_params, delete_existing_folder=True)

    def dump(self, wait=0, **keyvals):
        # HACK
        p = {"log_file": self.handles["log"].storage.absolute()}
        _, cpu_time = next(self.qtea_sim.observables.read_cpu_time_from_log("/", p))
        self.runtime_info(ext_cpu_time=cpu_time)
        # output_folder is never parameterized
        measures = self.qtea_sim.get_static_obs(p)
        proj_meas = measures.pop("projective_measurements")
        assert not proj_meas, "projective measurements unsupported"
        # TODO: can you QTEA people please pick a name and stick to it, for god's sake
        bond_ent = (
            measures.pop("bondentropy", None)
            or measures.pop("bond_entropy", None)
            or measures.pop("bond_entropy0", None)
        )
        if bond_ent:
            self.res["entropy_cut"] = list(bond_ent.keys())
            self.res["entropy_val"] = list(bond_ent.values())
        for k, v in measures.items():
            self.res[k] = self._unravel_observable(v)
        super().dump(wait, **keyvals)
