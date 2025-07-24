import numpy as np


def one_hot(keys):
    """Returns a (masked) 1-hot encoding."""
    uniq = np.unique(keys)
    mask = keys != uniq[:, np.newaxis]
    return uniq, np.ma.masked_array(~mask, mask)


def group_by(vals, keys):
    # from version in qcd's paper.py
    arg = np.argsort(keys, axis=None)
    # groupby is 6x slower already on small arrays (~1000 elements)
    keys, inds = np.unique(keys.flat[arg], return_index=True)
    vals = np.split(vals.flat[arg], inds[1:])
    return keys, vals
    # return np.array([[d, vs.mean()] for d, vs in zip(dist, vals)]).T
    # return np.array([[d, c.mean(), c.std(ddof=1)] for d, c in zip(dist, corr)]).T


def append_til_ndim(a, ndim):
    a = np.asanyarray(a)
    return np.expand_dims(a, tuple(range(a.ndim, ndim)))


def silent_squeeze(a, axis=None):
    a = np.ma.asanyarray(a)  # HACK otherwise squeeze destroys mask
    try:
        a = np.squeeze(a, axis=axis)
    except ValueError:
        if axis is None:
            raise
    return a


def tilepad(a, shape):
    a = np.asanyarray(a)
    dshape = np.array([a.shape, shape]) - a.shape
    return np.pad(a, dshape.T, mode="wrap")


def fftsymshift(dat, axis=None):
    dat = np.fft.fftshift(dat, axes=axis)
    if axis is None:
        axis = range(dat.ndim)
    pad = np.zeros((dat.ndim, 2), int)
    pad[list(axis), 1] = 1  # natively supports negative axis indices
    pad &= (np.expand_dims(dat.shape, 1) + 1) % 2
    dat = np.pad(dat, pad_width=pad, mode="wrap")
    return dat


def coords_to_slices(r):
    return [slice(c, c + 1) if c != -1 else slice(c, None) for c in r]
