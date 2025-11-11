import json
import matplotlib.pyplot as plt
import numpy as np
import time

import convert
import count
import der
import renorm
import utils

path = 'C:/Users/gubar/PycharmProjects/Navier'

Q = 16
K = 2
bound = 10
file_number = 4
viscosity = 0.001
tau = 0.01
incompressibility_mult = 1

points = Q ** K
sites = int(np.log2(Q))
d = 2 ** K

with open(f'{path}/data/result_{file_number}.json') as file:
    result = json.load(file)

vx, vy = utils.compr_in_data(np.array(result['vx']), np.array(result['vy']), Q)

cvx = convert.from_class_to_state(points, d, sites, K, vx)
cvy = convert.from_class_to_state(points, d, sites, K, vy)

mv = [convert.to_left_mps_with_svd(cvx, sites, d, bound), convert.to_left_mps_with_svd(cvy, sites, d, bound)]

for i in range(K):
    renorm.right_with_qr(mv[i])

h = 1 / Q

deriv = [der.x(sites, h), der.y(sites, h)]
deriv_2 = [der.x_2(sites, h), der.y_2(sites, h)]

columns = mv[0][0][0].shape[1]
rows = mv[0][0][0].shape[0]

start = time.time()
alpha, time_dict = count.new_node(d, rows, columns, mv, K, deriv, deriv_2,
                                   bound, viscosity, tau, incompressibility_mult)
end = time.time()

for sigma in range(d):
    for i in range(K):
        for row in range(rows):
            for column in range(columns):
                mv[i][0][sigma][row, column] = alpha[sigma, i, row, column]

total = end - start
print(f'Total time: {np.round(total, 1)} s')

for op in time_dict:
    time_op = time_dict[op]
    print(f'{op}: {np.round(time_op, 1)} s, {np.round(100 * time_op / total, 1)}%')

plt.imshow(convert.from_state_to_class(Q, points, d, sites, K, convert.from_mps(mv[0])))
plt.colorbar()

plt.xlabel('y')
plt.ylabel('x')

plt.show()
