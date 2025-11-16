import numpy as np

import renorm
import utils


def iteration(mps, guess_left, stop_diff, unitary=True):
    d = mps[0].shape[0]
    nodes = len(mps)
    prev_diff = 2
    diff = 1
    sum_trace = 0

    while prev_diff - diff > stop_diff:
        prev_diff = diff

        for i in reversed(range(1, nodes)):
            utils.update_guess(i, d, guess_left, mps, nodes)
            r, q = np.linalg.qr(np.conj(np.concatenate([guess_left[i][sigma] for sigma in range(d)], axis=1)).T)

            q = np.conj(q).T
            r = np.conj(r).T

            step = r.shape[1] // d

            for sigma in range(d):
                guess_left[i][sigma] = r[:, step * sigma: step * (sigma + 1)]
                guess_left[i - 1] = guess_left[i - 1] @ q

        for i in range(nodes - 1):
            utils.update_guess(i, d, guess_left, mps, nodes)
            site = guess_left[i]
            q, r = np.linalg.qr(site.reshape(-1, site[0].shape[1]))
            guess_left[i] = q.reshape(d, -1, q.shape[1])
            guess_left[i + 1] = r @ guess_left[i + 1]

        sum_trace = 0
        last_site = guess_left[nodes - 1]

        for sigma in range(d):
            m = last_site[sigma]
            sum_trace += np.trace(np.conj(m).T @ m)

        diff = 1 - sum_trace

    if unitary:
        guess_left[nodes - 1] /= np.sqrt(sum_trace)


def with_svd(mps_left, d_new, unitary=True):
    mps = []
    d = mps_left[0].shape[0]
    left = mps_left[-1]
    for i in reversed(range(len(mps_left))):
        u, s, b = np.linalg.svd(np.concatenate([left[sigma] for sigma in range(d)], axis=1), full_matrices=False)
        u = u[:, :d_new]
        norm = 1
        if not unitary:
            norm = np.linalg.norm(s)
        s = s[:d_new]
        s *= norm / np.linalg.norm(s)
        b = b[:d_new, :]
        step = b.shape[1] // d
        mps.append(np.zeros((d, b.shape[0], step)))
        for sigma in range(d):
            mps[-1][sigma] = b[:, step * sigma: step * (sigma + 1)]
        if i > 0:
            left = mps_left[i - 1] @ u @ np.diag(s)
        else:
            mps[-1] = u @ np.diag(s) @ mps[-1]
    return list(reversed(mps))


def mpo(mpo_tensors, d_new):
    sites = len(mpo_tensors)
    d = mpo_tensors[0].shape[0]

    for site in range(sites):
        oper = mpo_tensors[site]
        mpo_tensors[site] = oper.reshape(d * d, oper.shape[2], oper.shape[3])

    renorm.left_with_qr(mpo_tensors)
    compressed = with_svd(mpo_tensors, d_new, unitary=False)

    for site in range(sites):
        oper = compressed[site]
        mpo_tensors[site] = oper.reshape(d, d, oper.shape[1], oper.shape[2])

    return mpo_tensors
