import json
import matplotlib.pyplot as plt
import numpy as np
import sys

import convert
import count

path = 'C:/Users/gubar/PycharmProjects/Navier'

Q = 256
K = 2
bound = 7

d = 2 ** K
points = Q ** K
sites = int(np.log2(Q))
bonds = sites - 1

with open(f'{path}/data/result_4.json') as file:
    result = json.load(file)

vx = np.array(result['vx'])
vy = np.array(result['vy'])

cx = np.zeros(points)
cy = np.zeros(points)

for point in range(points):
    omega_array = count.count_sigma(point, d, sites)

    x = 0
    y = 0

    for j in range(sites):
        power = 2 ** (sites - j - 1)
        bit_array = count.count_sigma(omega_array[j], 2, K)

        x += bit_array[0] * power
        y += bit_array[1] * power

    cx[point] = vx[int(x), int(y)]
    cy[point] = vy[int(x), int(y)]

ax = convert.to_vidal(cx, sites, d)
max_lambda_amount = d ** ((bonds + 1) // 2)
lambda_array_x = np.zeros((bonds, max_lambda_amount))

for bond_idx in range(bonds):
    lambda_list_x = np.diag(ax[(bond_idx + 1) * 2])
    lambda_array_x[bond_idx, :len(lambda_list_x)] = lambda_list_x / np.linalg.norm(lambda_list_x)

ax_compr = convert.to_vidal(cx, sites, d, bound=bound)
ay_compr = convert.to_vidal(cy, sites, d, bound=bound)

cx_check = convert.from_mps(ax_compr)
cy_check = convert.from_mps(ay_compr)

vx_check = np.zeros((Q, Q))
vy_check = np.zeros((Q, Q))

for point in range(points):
    omega_array = count.count_sigma(point, d, sites)

    x = 0
    y = 0

    for j in range(sites):
        power = 2 ** (sites - j - 1)
        bit_array = count.count_sigma(omega_array[j], 2, K)

        x += bit_array[0] * power
        y += bit_array[1] * power

    vx_check[int(x), int(y)] = cx_check[point]
    vy_check[int(x), int(y)] = cy_check[point]

v = np.sqrt(vx ** 2 + vy ** 2)
v_check = np.sqrt(vx_check ** 2 + vy_check ** 2)
diff = np.linalg.norm(v - v_check) / np.linalg.norm(v)

mem = 0
for site in ax_compr:
    mem += sys.getsizeof(site)
mem /= sys.getsizeof(vx_check)

plt.imshow(lambda_array_x, aspect='auto', norm='log', extent=(0.5, max_lambda_amount + 0.5, bonds + 0.5, 0.5))
plt.colorbar(orientation='horizontal')
plt.title(r'$\frac{\lambda_{x \alpha}}{\sqrt{\sum_{\beta}{\lambda^{2}_{x \beta}}}}$')

plt.xticks(d ** np.arange(0, (bonds + 1) // 2 + 1))
plt.yticks(np.arange(1, sites))

plt.xlabel(r'$\alpha$')
plt.ylabel('Bond number')

plt.show()

plt.imshow(v)
plt.colorbar()
plt.title('v, dns solution')

plt.xlabel('y')
plt.ylabel('x')

plt.show()

plt.imshow(v_check)
plt.colorbar()
plt.title(f'v mps, diff = {np.round(100 * diff, 2)} %, mem = {np.round(100 * mem, 2)} %, D = {bound}')

plt.xlabel('y')
plt.ylabel('x')

plt.show()
