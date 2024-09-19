import argparse
import ctypes
import logging
import shlex
import signal
import sys
import time
from contextlib import contextmanager
from ctypes.util import find_library

import numpy as np

from simsio import rc
from simsio.utils import get_module_attr
from simsio.simulations import Simulation

__all__ = ["argparse", "run_sim"]

logger = logging.getLogger(__name__)

ARG_DELIM = "--"


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
    libraries = (
        "libiomp5.so",
        find_library("libiomp5md"),
        find_library("gomp"),
    )
    for l in libraries:
        if l is None:
            continue
        try:
            omp = ctypes.CDLL(l)
        except OSError:
            pass
        else:
            omp.omp_set_num_threads(int(num))
            return True

    logger.warning("OpenMP library not found: can't set nthreads")
    return False


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
def run_sim(sim_class=Simulation, not_found_ok=True, **sim_kwargs):
    # TODO: sim_class in rc
    delim = sys.argv.index(ARG_DELIM) if ARG_DELIM in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("group", type=str, help="group match pattern")
    parser.add_argument("uid", type=str, help="unique identifier of the simulation")
    parser.add_argument("ncores", type=int, help="number of CPU cores to use")
    parser.add_argument("--save-extras", action="store_true", help="save extras")
    args = parser.parse_args(args=sys.argv[1:delim])
    set_num_threads(args.ncores)
    if isinstance(sim_class, str):
        sim_class = get_module_attr(sim_class)
    sim_kwargs.setdefault("readonly", False)
    try:
        sim = sim_class.from_config(args.uid, args.group, **sim_kwargs)
    except KeyError as e:  # TODO: custom exception, missing config file or uid?
        if not_found_ok:
            logger.error(e.args[0])
            sys.exit()
        else:
            raise
    sim.ini_args = args
    sim.run_args = sys.argv[delim + 1 :]
    try:
        yield sim
        sim.dump()
        # TODO: delgate to scripts for specific extras
        # TODO: this does not work for handles previously removed from cache
        if not args.save_extras:
            for key in sim:
                if not key in rc["IO-handlers"]:
                    sim.unlink(key)
    except:
        logger.exception("Uncaught exception while running simulation")
        raise
    finally:
        sim.close()
