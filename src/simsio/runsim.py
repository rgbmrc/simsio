# ruff: noqa: F401 # TODO imports as reminders

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
from simsio.simulations import Simulation, default_sim_class
from simsio.utils import get_module_attr

__all__ = ["run_sim", "set_num_threads"]

logger = logging.getLogger(__name__)

ARG_DELIM = "--"


def set_num_threads(num):
    """Tries setting OpenMP number of threads to `num`.

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

    logger.warning("OpenMP library not found: can't set ncores")
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
def run_sim(sim_class=None, not_found_ok=True, **sim_kwargs):
    delim = sys.argv.index(ARG_DELIM) if ARG_DELIM in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("uid", type=str, help="unique identifier of the simulation")
    parser.add_argument("ncores", type=int, help="number of CPU cores to use")
    parser.add_argument(
        "-g",
        "--group",
        help="config group match pattern; without, `uid` runs from its stored par",
    )
    parser.add_argument(
        "--save-extras",
        action="store_true",
        help="write assets stashed with extra=True (skipped otherwise, never deleted)",
    )
    args = parser.parse_args(args=sys.argv[1:delim])
    set_num_threads(args.ncores)
    if sim_class is None:
        sim_class = default_sim_class()
    elif isinstance(sim_class, str):
        sim_class = get_module_attr(sim_class)
    # sim_kwargs.setdefault("readonly", False)  # NOTE should we allow readonly from_config?
    try:
        if args.group is None:  # e.g. par generated on another machine
            sim = sim_class(args.uid, readonly=False)
            logger.info("Running from the stored par, no config")
        else:
            sim = sim_class.from_config(args.uid, args.group, **sim_kwargs)
    # TODO: custom exception, missing config file or uid?
    except (KeyError, ValueError, FileNotFoundError) as e:
        if not_found_ok:
            logger.error(e.args[0])
            sys.exit()
        raise
    sim.ini_args = args
    sim.save_extras = args.save_extras
    sim.run_args = sys.argv[delim + 1 :]
    try:
        yield sim
        sim.dump()
    except:
        logger.exception("Uncaught exception while running simulation")
        raise
    finally:
        sim.close()
