import copy
import numpy as np

import count
import utils

def from_class_to_state(points, d, sites, dim, v):
    c = np.zeros(points)

    for point in range(points):
        omega_array = count.count_sigma(point, d, sites)

        x = 0
        y = 0

        for j in range(sites):
            power = 2 ** (sites - j - 1)
            bit_array = count.count_sigma(omega_array[j], 2, dim)

            x += bit_array[0] * power
            y += bit_array[1] * power

        x = int(x)
        y = int(y)

        c[point] = v[x, y]

    return c

def from_mps(a):
    a = copy.deepcopy(a)
    utils.eliminate_singular_matrix(a)
    d = a[0].shape[0]
    sites = len(a)
    n = d ** sites
    c = np.zeros(n)
    for i in range(n):
        sigma = count.count_sigma(i, d, sites)
        mps = np.eye(1)
        for j in range(sites):
            mps_new = mps @ a[j][int(sigma[j])]
            mps = mps_new
        c[i] = mps[0, 0]
    return c

def from_state_to_class(points_1_dim, points, d, sites, dim, c):
    v = np.zeros((points_1_dim, points_1_dim))

    for point in range(points):
        omega_array = count.count_sigma(point, d, sites)

        x = 0
        y = 0

        for j in range(sites):
            power = 2 ** (sites - j - 1)
            bit_array = count.count_sigma(omega_array[j], 2, dim)

            x += bit_array[0] * power
            y += bit_array[1] * power

        v[int(x), int(y)] = c[point]

    return v

def to_left_mps_with_qr(c, sites, d):
    steps = sites - 1
    a = []
    rank = 1
    psi = c.reshape(d, d ** (sites - 1))

    for i in range(steps):
        q, psi = np.linalg.qr(psi)
        rank_prev = rank
        rank = q.shape[1]
        a.append(np.zeros((d, rank_prev, rank)))

        for string in range(q.shape[0] // d):
            for sigma in range(d):
                a[i][sigma, string, :] = q[string * d + sigma, :]

        if i < steps - 1:
            psi = psi.reshape(d * rank, d ** (sites - i - 2))

    a.append(np.zeros((d, d, 1)))
    for sigma in range(d):
        a[steps][sigma, :, 0] = psi[:, sigma]

    return a

def to_left_mps_with_svd(c, sites, d, bound=None):
    steps = sites - 1
    a = []
    rank = 1
    psi = c.reshape(d, d ** (sites - 1))

    for i in range(steps):
        u, s, v = np.linalg.svd(psi, full_matrices=False)
        if bound is not None:
            u = u[:, :bound]
            s = s[:bound]
            v = v[:bound, :]
        rank_prev = rank
        rank = u.shape[1]
        a.append(np.zeros((d, rank_prev, rank)))

        for string in range(u.shape[0] // d):
            for sigma in range(d):
                a[i][sigma, string, :] = u[string * d + sigma, :]

        psi = np.diag(s) @ v
        if i < steps - 1:
            psi = psi.reshape(d * rank, d ** (sites - i - 2))

    a.append(np.zeros((d, rank, 1)))
    for sigma in range(d):
        a[steps][sigma, :, 0] = psi[:, sigma]

    return a

def to_vidal(c, sites, d, bound=None):
    steps = sites - 1
    mps = [np.eye(1)]
    rank = 1
    s = np.array([1])
    psi = c.reshape(d, d ** (sites - 1))
    eps = 1e-8

    for i in range(steps):
        s_prev = s
        u, s, v = np.linalg.svd(psi, full_matrices=False)
        if bound is not None:
            u = u[:, :bound]
            s = s[:bound]
            v = v[:bound, :]

        rank_prev = rank
        rank = u.shape[1]
        mps.append(np.zeros((d, rank_prev, rank)))

        for sigma in range(d):
            for string in range(u.shape[0] // d):
                mps[-1][sigma, string, :] = u[string * d + sigma, :]
        if np.min(s_prev) < eps:
            print('Warning!')
        mps[-1] = np.diag(1 / s_prev) @ mps[-1]

        lambda_matrix = np.diag(s)
        mps.append(lambda_matrix)
        psi = lambda_matrix @ v
        if i < steps - 1:
            psi = psi.reshape(d * rank, d ** (sites - i - 2))

    mps.append(np.zeros((d, rank, 1)))
    for sigma in range(d):
        mps[-1][sigma, :, 0] = psi[:, sigma]
    if np.min(s) < eps:
        print('Warning!')
    mps[-1] = np.diag(1 / s) @ mps[-1]
    mps.append(np.eye(1))
    return mps