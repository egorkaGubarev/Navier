import numpy as np

import utils


def left_with_qr(mps):
    d = mps[0].shape[0]
    utils.eliminate_singular_matrix(mps)
    sites = len(mps)
    for i in range(sites):
        site = mps[i]
        q, r = np.linalg.qr(site.reshape(-1, site[0].shape[1]))
        mps[i] = q.reshape(d, -1, q.shape[1])
        if i < sites - 1:
            mps[i + 1] = r @ mps[i + 1]
        else:
            mps[i] = mps[i] @ r


def right_with_qr(mps):
    d = mps[0].shape[0]
    utils.eliminate_singular_matrix(mps)
    for i in reversed(range(1, len(mps))):
        r, q = np.linalg.qr(np.conj(np.concatenate([mps[i][sigma] for sigma in range(d)], axis=1)).T)

        q = np.conj(q).T
        r = np.conj(r).T

        step = r.shape[1] // d
        new_site = np.zeros((d, r.shape[0], step))
        for sigma in range(d):
            new_site[sigma] = r[:, step * sigma: step * (sigma + 1)]

        mps[i] = new_site
        mps[i - 1] = mps[i - 1] @ q
