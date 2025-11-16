import copy
import numpy as np
import time

import compress
import renorm
import utils


def beta_vector(dim, sigma, mv_rep, alpha_1, beta, source):
    for i in range(dim):
        beta[sigma, i] = overlap_from_left(mv_rep[i], alpha_1[i]) - overlap_from_left(mv_rep[i], source[i])


def count_sigma(x, d, sites):
    sigma = np.zeros(sites)
    digit = 1
    while x > 0:
        sigma[sites - digit] = x % d
        x //= d
        digit += 1
    return sigma


def derivatives_for_step(dim, mv_rep, deriv, mv_der):
    for i in range(dim):
        mv_d = utils.apply(mv_rep[i], deriv[i])
        mv_der.append(mv_d)


def h_matrix(dim, h_matr, m_der, sigma):
    for i in range(dim):
        for j in range(dim):
            if j >= i:
                h_matr[sigma, i, j] = overlap_from_left(m_der[i], m_der[j])
            else:
                h_matr[sigma, i, j] = h_matr[sigma, j, i]


def new_node(d, rows, columns, mv, mv_alpha, dim, deriv, tau, incompressibility_mult, node, source):
    h_matr = np.zeros((d, dim, dim))
    beta = np.zeros((d, dim))
    alpha = np.zeros((d, dim, rows, columns))

    time_rep_node = 0
    time_der = 0
    time_h = 0
    time_b = 0

    for sigma in range(d):
        for row in range(rows):
            for column in range(columns):
                mv_der = []

                rep_node_start = time.time()
                mv_rep = vel_with_rep_node(mv, dim, rows, columns, sigma, d, row, column, node)
                rep_node_end = time.time()
                time_rep_node += rep_node_end - rep_node_start

                der_start = time.time()
                derivatives_for_step(dim, mv_rep, deriv, mv_der)
                der_end = time.time()
                time_der += der_end - der_start

                h_start = time.time()
                h_matrix(dim, h_matr, mv_der, sigma)
                h_end = time.time()
                time_h += h_end - h_start

                b_start = time.time()
                beta_vector(dim, sigma, mv_rep, mv_alpha, beta, source)
                b_end = time.time()
                time_b += b_end - b_start

                alpha[sigma, :, row, column] = utils.cgd((np.eye(dim) -
                                                          incompressibility_mult * tau ** 2 * h_matr[sigma]),
                                                         beta[sigma])[-1]

    return alpha, {'rep node': time_rep_node, 'der': time_der, 'comp h': time_h,
                   'comp b': time_b}


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


def source_term(dim, mv_beta, deriv, bound, deriv_2, viscosity, tau):
    source = []

    for i in range(dim):
        source_i = None

        for j in range(dim):
            der_in_beta = utils.apply(mv_beta[i], deriv[j])
            renorm.left_with_qr(der_in_beta)
            der_in_beta = compress.with_svd(der_in_beta, bound, unitary=False)
            nonlin = utils.mult(mv_beta[j], der_in_beta)
            viscosity_term = utils.apply(mv_beta[i], deriv_2[j])
            viscosity_term[0] *= viscosity
            source_i_j = utils.add(nonlin, viscosity_term)

            if source_i is None:
                source_i = source_i_j
            else:
                source_i = utils.add(source_i, source_i_j)

        source_i[0] *= tau
        renorm.left_with_qr(source_i)
        source_i = compress.with_svd(source_i, bound, unitary=False)
        source.append(source_i)

    return source


def vel_with_rep_node(mv, dim, rows, columns, sigma, d, row, column, node):
    mv_rep = copy.deepcopy(mv)

    for i in range(dim):
        mv_rep[i][node] = np.zeros((d, rows, columns))
        mv_rep[i][node][sigma][row, column] = 1

    return mv_rep
