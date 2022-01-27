import logging
import signal
import sys
import time
from contextlib import contextmanager

import numpy as np

# TODO: remove tenpy dependencies
from tenpy.tools.events import EventHandler
from tenpy.tools.process import mkl_set_nthreads, omp_set_nthreads

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
        nthreads = int(nthreads)
        omp_set_nthreads(nthreads) or mkl_set_nthreads(nthreads)
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
        sim["par"].warn_unused(recursive=True)
        sim.close()
