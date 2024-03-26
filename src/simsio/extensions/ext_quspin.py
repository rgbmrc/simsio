from math import log

import numpy as np
from quspin.operators import hamiltonian


class Observable:

    def __init__(self, terms, sites):
        self.terms = terms
        self.sites = sites

    def __call__(self, state, time=0):
        evs = [t.expt_value(state, time) for t in self.terms]
        shape = self.sites.shape * int(log(len(evs), self.sites.size))
        return np.reshape(evs, shape)


class ObsLocal(Observable):

    def __init__(self, terms, sites, **obs_kwargs):
        terms = [
            hamiltonian([[o, [[v, s]]] for o, v in terms.items()], [], **obs_kwargs)
            for s in sites.flat
        ]
        super().__init__(terms, sites)

    # NOTE: if needed override
    # def __call__(self, state, time=0):
    #     evs = [t.expt_value(state) for t in self.terms]
    #     return np.reshape(evs, self.sites.shape)


def make_observables(measures_pars, **common_kwargs):
    observables = {}
    for name, (cls, args) in measures_pars.items():
        cls = globals()[cls]  # HACK: use module instead?
        observables[name] = cls(*args, **common_kwargs)
    return observables
