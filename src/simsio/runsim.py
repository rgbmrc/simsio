import ctypes
import logging
import signal
import sys
import time
from contextlib import contextmanager
from ctypes.util import find_library

import numpy as np

# TODO: remove tenpy dependencies
from tenpy.tools.events import EventHandler


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


from simsio.simulations import Simulation

logger = logging.getLogger(__name__)

shelve = False
checkpoint = EventHandler()


def set_shelve(signum, frame):
    global shelve
    shelve = True
    logger.warning(f"Signal {signum} received: shelving at checkpoint")


def checkpointed(iterable):
    for i in iterable:
        logger.info("checkpoint after iteration step completed")
        checkpoint.emit()
        yield i


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
def run_sim(**sim_kwargs):
    # TODO: generalize (or delegate) args parsing
    try:
        signal.signal(signal.SIGUSR1, set_shelve)
        group, uid, nthreads = sys.argv[1:]
        set_num_threads(int(nthreads))
        sim_kwargs.setdefault("readonly", False)
        sim = Simulation(uid, group, **sim_kwargs)
        checkpoint.connect(sim.dump)
    except:
        logger.exception("Uncaught exception while loading simulation")
        raise

    try:
        yield sim
    except:
        logger.exception("Uncaught exception while running simulation")
        raise
    finally:
        sim.dump()
        if hasattr(sim["par"], "warn_unused"):
            sim["par"].warn_unused(recursive=True)
        sim.close()
