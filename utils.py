import numpy as np
import scipy
import time

import count
import renorm


def add(mps_1, mps_2):
    mps = []

    site_1 = mps_1[0]
    site_2 = mps_2[0]

    d = site_1.shape[0]
    mps.append(np.zeros((d, 1, site_1.shape[2] + site_2.shape[2])))
    for sigma in range(d):
        mps[0][sigma][0, :] = np.concatenate((site_1[sigma][0, :], site_2[sigma][0, :]))
    for site_idx in range(1, len(mps_1) - 1):
        site_1 = mps_1[site_idx]
        site_2 = mps_2[site_idx]

        rows = site_1.shape[-2] + site_2.shape[-2]
        columns = site_1.shape[-1] + site_2.shape[-1]
        if len(site_1.shape) == 3:
            mps.append(np.zeros((d, rows, columns)))
            for sigma in range(d):
                mps[site_idx][sigma] = scipy.linalg.block_diag(site_1[sigma], site_2[sigma])
        else:
            mps.append(np.zeros((rows, columns)))
            mps[site_idx] = scipy.linalg.block_diag(site_1, site_2)

    site_1 = mps_1[-1]
    site_2 = mps_2[-1]

    mps.append(np.zeros((d, site_1.shape[1] + site_2.shape[1], 1)))
    for sigma in range(d):
        mps[-1][sigma][:, 0] = np.concatenate((site_1[sigma][:, 0], site_2[sigma][:, 0]))
    return mps


def add_mpo(mpo_1, mpo_2):
    mpo = []
    site_1 = mpo_1[0]
    d = site_1.shape[0]
    mpo.append(np.concatenate((site_1[:, :, 0, :], mpo_2[0][:, :, 0, :]), axis=-1).reshape(d, d, 1, -1))

    for site_idx in range(1, len(mpo_1) - 1):
        mpo.append(scipy.linalg.block_diag(mpo_1[site_idx][:, :], mpo_2[site_idx][:, :]))

    mpo.append(np.concatenate((mpo_1[-1][:, :, :, 0], mpo_2[-1][:, :, :, 0]), axis=-1).reshape(d, d, -1, 1))
    return mpo


def apply(mps, mpo):
    result = []
    d = mps[0].shape[0]

    for site in range(len(mps)):
        m = mps[site]
        b = mpo[site]

        matrix = m[0]
        oper = b[0][0]
        result.append(np.einsum('stab,tij->saibj',
                                b, m).reshape(d, oper.shape[0] * matrix.shape[0],
                                              oper.shape[1] * matrix.shape[1]))

    return result


def cgd(a, b):
    dim = a.shape[0]
    x = np.random.randn(dim)
    r = b - a @ x
    d = r
    rr = r @ r
    xs = [x]

    for i in range(1, dim):
        ad = a @ d
        alpha = rr / (d @ ad)
        x = x + alpha * d
        r = r - alpha * ad
        rr_new = r @ r
        beta = rr_new / rr
        d = r + beta * d
        rr = rr_new

        xs.append(x)

    return np.array(xs)


def compr_in_data(vx_full, vy_full, new_points):
    vx = np.zeros((new_points, new_points))
    vy = np.zeros((new_points, new_points))

    compr = int(len(vx_full) / new_points)

    for i in range(new_points):
        i_new = i * compr

        for j in range(new_points):
            vx[i, j] = vx_full[i_new, j * compr]
            vy[i, j] = vy_full[i_new, j * compr]

    return vx, vy


def eliminate_singular_matrix(mps):
    if len(mps[0].shape) == 2:
        del mps[0]
    if len(mps[-1].shape) == 2:
        del mps[-1]
    i = 0
    while i < len(mps):
        site = mps[i]
        if len(site.shape) == 2:
            mps[i - 1] = mps[i - 1] @ site
            del mps[i]
        else:
            i += 1


def ident(sites, d):
    mpo = []
    bulk = np.zeros((d, d, 1, 1))
    bulk[:, :, 0, 0] = np.eye(d)

    for site in range(sites):
        mpo.append(bulk)

    return mpo


def make_runge_step(dim, mv_beta, deriv, bound, deriv_2, viscosity, dt,
                    sweeps, sites, d, mv_opt, mv_alpha, incompressibility_mult, time_dict):
    source = count.source_term(dim, mv_beta, deriv, bound, deriv_2, viscosity, dt)

    for sweep_idx in range(sweeps):
        time_dict = sweep(sites, d, mv_opt, mv_alpha, dim, deriv, dt, incompressibility_mult, time_dict, source)
        time_dict = sweep_back(sites, d, mv_opt, mv_alpha, dim, deriv, dt, incompressibility_mult, time_dict, source)

    return time_dict


def mult(mps1, mps2):
    mps = []
    d = mps1[0].shape[0]

    for site_idx in range(len(mps1)):
        site1 = mps1[site_idx]
        site2 = mps2[site_idx]

        matrix1 = site1[0]
        matrix2 = site2[0]

        mps.append(np.einsum('sij,skl->sikjl',
                             site1, site2).reshape(d, matrix1.shape[0] * matrix2.shape[0],
                                                   matrix1.shape[1] * matrix2.shape[1]))

    return mps


def mult_mpo(mpo1, mpo2):
    result = []
    for site_id in range(len(mpo1)):
        site1 = mpo1[site_id]
        site2 = mpo2[site_id]

        matrix1 = site1[0, 0]
        matrix2 = site2[0, 0]

        d = site1.shape[0]

        rows1 = matrix1.shape[0]
        rows2 = matrix2.shape[0]
        columns1 = matrix1.shape[1]
        columns2 = matrix2.shape[1]

        n = np.zeros((d, d, rows1 * rows2, columns1 * columns2))

        for a1 in range(rows1):
            for a2 in range(rows2):
                for b1 in range(columns1):
                    for b2 in range(columns2):
                        for sigma1 in range(d):
                            for sigma2 in range(d):
                                for sigma_sum in range(d):
                                    n[sigma1, sigma2][a1 * rows2 + a2, b1 * columns2 + b2] += (
                                            site1[sigma1, sigma_sum, a1, b1] * site2[sigma_sum, sigma2, a2, b2])

        result.append(n)
    return result


def sweep(sites, d, mv, mv_alpha, dim, deriv, tau, incompressibility_mult, time_dict, source):
    for i in range(dim):
        renorm.right_with_qr(mv[i])

    for node in range(sites):
        time_dict_new = update_node(d, mv, mv_alpha, dim, deriv, tau, incompressibility_mult, node, source)

        for i in range(dim):
            site = mv[i][node]
            q, r = np.linalg.qr(site.reshape(-1, site[0].shape[1]))
            mv[i][node] = q.reshape(d, -1, q.shape[1])
            if node < sites - 1:
                mv[i][node + 1] = r @ mv[i][node + 1]
            else:
                mv[i][node] = mv[i][node] @ r

        for op in time_dict_new:
            if op in time_dict:
                time_dict_new[op] += time_dict[op]

        time_dict = time_dict_new

    return time_dict


def sweep_back(sites, d, mv, mv_alpha, dim, deriv, tau, incompressibility_mult, time_dict, source):
    for node in reversed(range(sites)):
        time_dict_new = update_node(d, mv, mv_alpha, dim, deriv, tau, incompressibility_mult, node, source)

        for i in range(dim):
            r, q = np.linalg.qr(np.conj(np.concatenate([mv[i][node][sigma] for sigma in range(d)], axis=1)).T)

            q = np.conj(q).T
            r = np.conj(r).T

            step = r.shape[1] // d
            new_site = np.zeros((d, r.shape[0], step))
            for sigma in range(d):
                new_site[sigma] = r[:, step * sigma: step * (sigma + 1)]

            mv[i][node] = new_site
            mv[i][node - 1] = mv[i][node - 1] @ q

        for op in time_dict_new:
            if op in time_dict:
                time_dict_new[op] += time_dict[op]

        time_dict = time_dict_new

    return time_dict


def update_guess(i, d, guess, mps, nodes):
    left = np.eye(1)
    right = np.eye(1)

    for site in range(i):
        core = 0
        for sigma in range(d):
            core += np.conj(guess[site][sigma]).T @ left @ mps[site][sigma]
        left = core
    for site in reversed(range(i + 1, nodes)):
        core = 0
        for sigma in range(d):
            core += mps[site][sigma] @ right @ np.conj(guess[site][sigma]).T
        right = core
    guess[i] = left @ mps[i] @ right


def update_node(d, mv, mv_alpha, dim, deriv, tau, incompressibility_mult, node, source):
    columns = mv[0][node][0].shape[1]
    rows = mv[0][node][0].shape[0]

    start = time.time()
    alpha, time_dict = count.new_node(d, rows, columns, mv, mv_alpha, dim,
                                      deriv, tau, incompressibility_mult, node, source)

    for i in range(dim):
        mv[i][node][:][:, :] = alpha[:, i, :, :]

    time_dict.update({'total': time.time() - start})
    return time_dict
