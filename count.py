import copy
import numpy as np
import time

import compress
import renorm
import utils

def beta_vector(dim, sigma, mv_rep, alpha_1, alpha_2, beta, deriv, deriv_2, viscosity, tau, bound):
    time_nonlin = 0
    time_had = 0
    time_der = 0
    mpo_time = 0
    renorm_time = 0

    for i in range(dim):
        sec = 0

        for j in range(dim):
            start = time.time()

            start_had = time.time()

            start_der = time.time()

            start_mpo = time.time()
            der = utils.apply(alpha_2[i], deriv[j])
            mpo_time += time.time() - start_mpo

            start_renorm = time.time()
            renorm.left_with_qr(der)
            renorm_time += time.time() - start_renorm

            der = compress.with_svd(der, bound, unitary=False)
            time_der += time.time() - start_der

            nonlin = utils.mult(alpha_2[j], der)
            time_had += time.time() - start_had

            sec += overlap_from_left(mv_rep[i], nonlin)
            time_nonlin += time.time() - start

            sec += viscosity * overlap_from_left(mv_rep[i], utils.apply(alpha_2[i], deriv_2[j]))

        beta[sigma, i] = overlap_from_left(mv_rep[i], alpha_1[i]) - tau * sec

    return {'count nonlin': time_nonlin, 'had': time_had, 'der': time_der, 'mpo': mpo_time, 'renorm': renorm_time}

def count_sigma(x, d, sites):
    sigma = np.zeros(sites)
    digit = 1
    while x > 0:
        sigma[sites - digit] = x % d
        x //= d
        digit += 1
    return sigma

def derivatives_for_step(dim, mv_rep, deriv, mv_der, bound):
    for i in range(dim):
        mv_d = utils.apply(mv_rep[i], deriv[i])
        renorm.left_with_qr(mv_d)
        mv_der.append(compress.with_svd(mv_d, bound, unitary=False))

def h_matrix(dim, h_matr, m_der, sigma):
    for i in range(dim):
        for j in range(dim):
            if j >= i:
                h_matr[sigma, i, j] = overlap_from_left(m_der[i], m_der[j])
            else:
                h_matr[sigma, i, j] = h_matr[sigma, j, i]

def new_node(d, rows, columns, mv, dim, deriv, deriv_2,
             bound, viscosity, tau, incompressibility_mult):
    h_matr = np.zeros((d, dim, dim))
    beta = np.zeros((d, dim))
    alpha = np.zeros((d, dim, rows, columns))

    time_rep_node = 0
    time_der = 0
    time_h = 0
    time_b = 0
    time_nonlin = 0
    time_had = 0
    time_der_in_had = 0
    mpo_time = 0
    renorm_time = 0

    for sigma in range(d):
        for row in range(rows):
            for column in range(columns):
                mv_der = []

                rep_node_start = time.time()
                mv_rep = vel_with_rep_node(mv, dim, columns, sigma, d, row, column)
                rep_node_end = time.time()
                time_rep_node += rep_node_end - rep_node_start

                der_start = time.time()
                derivatives_for_step(dim, mv_rep, deriv, mv_der, bound)
                der_end = time.time()
                time_der += der_end - der_start

                h_start = time.time()
                h_matrix(dim, h_matr, mv_der, sigma)
                h_end = time.time()
                time_h += h_end - h_start

                b_start = time.time()
                times = beta_vector(dim, sigma, mv_rep, mv, mv, beta, deriv, deriv_2, viscosity, tau, bound)
                b_end = time.time()
                time_b += b_end - b_start

                time_nonlin += times['count nonlin']
                time_had += times['had']
                time_der_in_had += times['der']
                mpo_time += times['mpo']
                renorm_time + times['renorm']

                alpha[sigma, :, row, column] = utils.cgd((np.eye(dim) -
                                                          incompressibility_mult * tau ** 2 * h_matr[sigma]),
                                                         beta[sigma])[-1]

    return alpha, {'rep node': time_rep_node, 'der': time_der, 'comp h': time_h,
                   'comp b': time_b, 'count nonlin': time_nonlin,
                   'count had': time_had, 'count der in had': time_der_in_had,
                   'apply mpo': mpo_time, 'renorm': renorm_time}

def overlap_from_left(mps_1, mps_2):
    d = mps_1[0].shape[0]
    over = np.eye(1)
    for site in range(len(mps_1)):
        if len(mps_1[site].shape) == 3:
            core = 0
            for sigma in range(d):
                core += np.conjugate(mps_2[site][sigma]).T @ over @ mps_1[site][sigma]

            over = core
        else:
            over = np.conjugate(mps_2[site]).T @ over @ mps_1[site]
    return np.array(over)[0][0]

def vel_with_rep_node(mv, dim, columns, sigma, d, row, column):
    mv_rep = copy.deepcopy(mv)

    for i in range(dim):
        mv_rep[i][0] = np.zeros((d, 1, columns))
        mv_rep[i][0][sigma][row, column] = 1

    return mv_rep
