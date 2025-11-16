import compress
import shift
import utils


def x(sites, h):
    right = shift.right_x(sites)
    right[0] *= -1
    der_mpo = utils.add_mpo(shift.left_x(sites), right)
    der_mpo[0] /= 2 * h
    return compress.mpo(der_mpo, 3)


def x_4_ord(sites, h):
    site_m_1 = shift.right_x(sites)
    site_1 = shift.left_x(sites)

    site_m_2 = utils.mult_mpo(site_m_1, site_m_1)
    site_2 = utils.mult_mpo(site_1, site_1)

    site_m_2[0] /= 12 * h
    site_2[0] /= -12 * h

    site_m_1[0] *= - 2 / (3 * h)
    site_1[0] *= 2 / (3 * h)

    return compress.mpo(utils.add_mpo(utils.add_mpo(site_m_2, site_m_1), utils.add_mpo(site_1, site_2)), 3)


def x_8_ord(sites, h):
    site_m_1 = shift.right_x(sites)
    site_1 = shift.left_x(sites)

    site_m_2 = utils.mult_mpo(site_m_1, site_m_1)
    site_2 = utils.mult_mpo(site_1, site_1)

    site_m_3 = utils.mult_mpo(site_m_2, site_m_1)
    site_3 = utils.mult_mpo(site_2, site_1)

    site_m_4 = utils.mult_mpo(site_m_2, site_m_2)
    site_4 = utils.mult_mpo(site_2, site_2)

    site_m_4[0] /= 280 * h
    site_4[0] /= -280 * h

    site_m_3[0] *= - 4 / (105 * h)
    site_3[0] *= 4 / (105 * h)

    site_m_2[0] /= 5 * h
    site_2[0] /= -5 * h

    site_m_1[0] *= - 4 / (5 * h)
    site_1[0] *= 4 / (5 * h)

    return compress.mpo(utils.add_mpo(utils.add_mpo(utils.add_mpo(site_m_2, site_m_1), utils.add_mpo(site_1, site_2)),
                                      utils.add_mpo(utils.add_mpo(site_m_4, site_m_3), utils.add_mpo(site_3, site_4))),
                        3)


def y(sites, h):
    right = shift.right_y(sites)
    right[0] *= -1
    der_mpo = utils.add_mpo(shift.left_y(sites), right)
    der_mpo[0] /= 2 * h
    return compress.mpo(der_mpo, 3)


def x_2(sites, h):
    center = utils.ident(sites, 4)
    center[0] = center[0] * -2
    der_mpo = utils.add_mpo(utils.add_mpo(shift.left_x(sites), center), shift.right_x(sites))
    der_mpo[0] /= h ** 2
    return compress.mpo(der_mpo, 3)


def y_2(sites, h):
    center = utils.ident(sites, 4)
    center[0] = center[0] * -2
    der_mpo = utils.add_mpo(utils.add_mpo(shift.left_y(sites), center), shift.right_y(sites))
    der_mpo[0] /= h ** 2
    return compress.mpo(der_mpo, 3)
