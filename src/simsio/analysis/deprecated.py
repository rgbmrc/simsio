import re
import dpath
from simsio.simulations import sim_or_uid_arg


@sim_or_uid_arg
def extract_text(sim, key, regex, reverse=False, op="search"):
    d = sim[key]
    if reverse:
        d = "\n".join(reversed(d.splitlines()))
    return getattr(re.compile(regex), op)(d)


@sim_or_uid_arg
def extract_dict(sim, key, glob, op=None):
    op = op or dpath.get
    return op(sim[key], glob)
