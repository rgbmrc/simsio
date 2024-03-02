import argparse
import ctypes
import logging
import signal
import sys
import time
from contextlib import contextmanager
from ctypes.util import find_library

import numpy as np


def set_num_threads(num):
    """
    Tries setting OpenMP number of threads to `num`.

    Parameters
    ----------
    num : int

    Returns
    -------
    success : bool
    """
    for l in ["libiomp5.so", find_library("libiomp5md"), find_library("gomp")]:
        if l is None:
            continue
        try:
            omp = ctypes.CDLL(l)
        except OSError:
            pass
        else:
            omp.omp_set_num_threads(int(num))
            return True

    logger.error("OpenMP library not found: can't set nthreads")
    return False


from simsio import rc
from simsio.simulations import Simulation
from simsio.iocore import _get_mod_attr

logger = logging.getLogger(__name__)


def build_measures(measures, **context):
    # TODO: handle special notation for expectation values & correlation functions?
    return {k: m if callable(m) else eval(m, context) for k, m in measures.items()}


def append_measures(measures, results, target=None):
    results |= {k: list(results.get(k, [])) for k in measures}
    (samples,) = {len(results[k]) for k in measures}
    while not target or samples < target:
        if samples:
            yield samples
        for k, msr in measures.items():
            try:
                res = np.asanyarray(msr())
            except:
                logger.error(f"Unable to measure {k}")
                raise
            results[k].append(res)
        samples += 1


@contextmanager
def run_sim(sim_class=Simulation, **sim_kwargs):
    # TODO: delegate arg parsing
    # TODO: sim_class in rc
    parser = argparse.ArgumentParser()
    parser.add_argument("group", type=str, help="group match pattern")
    parser.add_argument("uid", type=str, help="unique identifier of the simulation")
    parser.add_argument("ncores", type=int, help="number of CPU cores to use")
    parser.add_argument("--save-extras", action="store_true", help="save extras")
    args = parser.parse_args()
    try:
        if isinstance(sim_class, str):
            sim_class = _get_mod_attr(sim_class)
        set_num_threads(args.ncores)
        sim_kwargs.setdefault("readonly", False)
        sim = sim_class(args.uid, args.group, **sim_kwargs)
        sim.run_args = args
    except:
        logger.exception("Uncaught exception while loading simulation")
        raise

    try:
        yield sim
        sim.dump()
        # TODO: delgate to scripts for specific extras
        # TODO: this does not work for handles that have been previously removed from cache
        if not args.save_extras:
            for key in sim:
                if not key in rc["IO-handlers"]:
                    sim.unlink(key)
    except:
        logger.exception("Uncaught exception while running simulation")
        raise
    finally:
        if hasattr(sim["par"], "warn_unused"):
            sim["par"].warn_unused(recursive=True)
        sim.close()
