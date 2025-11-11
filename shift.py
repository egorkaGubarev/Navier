import numpy as np

import compress
import utils


def left_x(sites):
    bulk = np.zeros((2, 2, 2, 2))
    first = np.zeros((2, 2, 1, 2))
    last = np.zeros((2, 2, 2, 1))

    bulk[0, 0, 0, 0] = 1
    bulk[0, 1, 0, 1] = 1
    bulk[1, 0, 1, 1] = 1
    bulk[1, 1, 0, 0] = 1

    first[0, 0, 0, 0] = 1
    first[0, 1, 0, 1] = 1
    first[1, 0, 0, 1] = 1
    first[1, 1, 0, 0] = 1

    last[0, 1, 0, 0] = 1
    last[1, 0, 1, 0] = 1

    bulk_eye = np.zeros((2, 2, 2, 2))

    for sigma in range(2):
        bulk_eye[sigma, sigma, :, :] = np.eye(2)

    first_eye = bulk_eye[:, :, 0, :].reshape(2, 2, 1, 2)
    last_eye = bulk_eye[:, :, :, 0].reshape(2, 2, 2, 1)

    bulk = np.kron(bulk, bulk_eye)
    first = np.kron(first, first_eye)
    last = np.kron(last, last_eye)

    mpo = [first]

    for site in range(sites - 2):
        mpo.append(bulk)

    mpo.append(last)
    return compress.mpo(mpo, 2)

def left_2x(sites):
    return utils.mult_mpo(left_x(sites), left_x(sites))

def left_y(sites):
    bulk = np.zeros((2, 2, 2, 2))
    first = np.zeros((2, 2, 1, 2))
    last = np.zeros((2, 2, 2, 1))

    bulk[0, 0, 0, 0] = 1
    bulk[0, 1, 0, 1] = 1
    bulk[1, 0, 1, 1] = 1
    bulk[1, 1, 0, 0] = 1

    first[0, 0, 0, 0] = 1
    first[0, 1, 0, 1] = 1
    first[1, 0, 0, 1] = 1
    first[1, 1, 0, 0] = 1

    last[0, 1, 0, 0] = 1
    last[1, 0, 1, 0] = 1

    bulk_eye = np.zeros((2, 2, 2, 2))

    for sigma in range(2):
        bulk_eye[sigma, sigma, :, :] = np.eye(2)

    first_eye = bulk_eye[:, :, 0, :].reshape(2, 2, 1, 2)
    last_eye = bulk_eye[:, :, :, 0].reshape(2, 2, 2, 1)

    bulk = np.kron(bulk_eye, bulk)
    first = np.kron(first_eye, first)
    last = np.kron(last_eye, last)

    mpo = [first]

    for site in range(sites - 2):
        mpo.append(bulk)

    mpo.append(last)
    return compress.mpo(mpo, 2)

def right_x(sites):
    bulk = np.zeros((2, 2, 2, 2))
    first = np.zeros((2, 2, 1, 2))
    last = np.zeros((2, 2, 2, 1))

    bulk[0, 0, 0, 0] = 1
    bulk[1, 0, 0, 1] = 1
    bulk[0, 1, 1, 1] = 1
    bulk[1, 1, 0, 0] = 1

    first[0, 0, 0, 0] = 1
    first[0, 1, 0, 1] = 1
    first[1, 0, 0, 1] = 1
    first[1, 1, 0, 0] = 1

    last[1, 0, 0, 0] = 1
    last[0, 1, 1, 0] = 1

    bulk_eye = np.zeros((2, 2, 2, 2))

    for sigma in range(2):
        bulk_eye[sigma, sigma, :, :] = np.eye(2)

    first_eye = bulk_eye[:, :, 0, :].reshape(2, 2, 1, 2)
    last_eye = bulk_eye[:, :, :, 0].reshape(2, 2, 2, 1)

    bulk = np.kron(bulk, bulk_eye)
    first = np.kron(first, first_eye)
    last = np.kron(last, last_eye)

    mpo = [first]

    for site in range(sites - 2):
        mpo.append(bulk)

    mpo.append(last)
    return compress.mpo(mpo, 2)

def right_y(sites):
    bulk = np.zeros((2, 2, 2, 2))
    first = np.zeros((2, 2, 1, 2))
    last = np.zeros((2, 2, 2, 1))

    bulk[0, 0, 0, 0] = 1
    bulk[1, 0, 0, 1] = 1
    bulk[0, 1, 1, 1] = 1
    bulk[1, 1, 0, 0] = 1

    first[0, 0, 0, 0] = 1
    first[0, 1, 0, 1] = 1
    first[1, 0, 0, 1] = 1
    first[1, 1, 0, 0] = 1

    last[1, 0, 0, 0] = 1
    last[0, 1, 1, 0] = 1

    bulk_eye = np.zeros((2, 2, 2, 2))

    for sigma in range(2):
        bulk_eye[sigma, sigma, :, :] = np.eye(2)

    first_eye = bulk_eye[:, :, 0, :].reshape(2, 2, 1, 2)
    last_eye = bulk_eye[:, :, :, 0].reshape(2, 2, 2, 1)

    bulk = np.kron(bulk_eye, bulk)
    first = np.kron(first_eye, first)
    last = np.kron(last_eye, last)

    mpo = [first]

    for site in range(sites - 2):
        mpo.append(bulk)

    mpo.append(last)
    return compress.mpo(mpo, 2)
