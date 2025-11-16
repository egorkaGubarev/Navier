import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import convert
import der
import utils

path = 'D:/PycharmProjects/Navier'

Q = 256
K = 2
bound = 10
file_number = 0
viscosity = 0.001
tau = 0.001
incompressibility_mult = 1
time_precis = 2
steps = 3
sweeps = 1

points = Q ** K
sites = int(np.log2(Q))
d = 2 ** K
h = 1 / Q
dt = tau / 2

with open(f'{path}/data/result_{file_number}.json') as file:
    result = json.load(file)

vx, vy = utils.compr_in_data(np.array(result['vx']), np.array(result['vy']), Q)

cvx = convert.from_class_to_state(points, d, sites, K, vx)
cvy = convert.from_class_to_state(points, d, sites, K, vy)

mv = [convert.to_left_mps_with_svd(cvx, sites, d, bound), convert.to_left_mps_with_svd(cvy, sites, d, bound)]

deriv = [der.x(sites, h), der.y(sites, h)]
deriv_2 = [der.x_2(sites, h), der.y_2(sites, h)]

time_dict = {}

for step in tqdm.tqdm(range(steps)):
    mv_alpha = copy.deepcopy(mv)
    mv_beta = copy.deepcopy(mv)
    mv_opt = copy.deepcopy(mv)

    time_dict = utils.make_runge_step(K, mv_beta, deriv, bound, deriv_2, viscosity, dt,
                                      sweeps, sites, d, mv_opt, mv_alpha, incompressibility_mult, time_dict)
    mv_next = copy.deepcopy(mv_opt)
    time_dict = utils.make_runge_step(K, mv_opt, deriv, bound, deriv_2, viscosity, tau,
                                      sweeps, sites, d, mv_next, mv_alpha, incompressibility_mult, time_dict)
    mv = mv_next

if steps > 0:
    total = time_dict['total']

    for op in time_dict:
        time_op = time_dict[op]
        print(f'{op}: {np.round(time_op, time_precis)} s, {np.round(100 * time_op / total, 1)}%')

plt.imshow(convert.from_state_to_class(Q, points, d, sites, K, convert.from_mps(mv[0])))
plt.colorbar()

plt.xlabel('y')
plt.ylabel('x')

plt.show()
