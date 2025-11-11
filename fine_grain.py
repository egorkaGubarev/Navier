import json
import matplotlib.pyplot as plt
import numpy as np

import convert
import count

path = 'C:/Users/gubar/PycharmProjects/Navier'

Q = 128
K = 2
bound = None
file_number = 4
c = 3

d = 2 ** K
points = Q ** K
sites = int(np.log2(Q))
bonds = sites - 1
vx = np.zeros((Q, Q))
cx = np.zeros(points)

with open(f'{path}/data/result_{file_number}.json') as file:
    result = json.load(file)

vx_full = np.array(result['vx'])
compr = int(len(vx_full) / Q)

for i in range(Q):
    i_new = i * compr

    for j in range(Q):
        vx[i, j] = vx_full[i_new, j * compr]

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

ax = convert.to_left_mps_with_svd(cx, sites, d, bound=bound)
a = []
l = np.zeros((1, c))
b = np.zeros((d, d, c, c))
r = np.zeros((c, 1))

l[0, 0] = 1
l[0, 1] = 1
l[0, 2] = 1

b[0, 0, 0, 0] = 1
b[1, 1, 0, 0] = 1
b[2, 2, 0, 0] = 1
b[3, 3, 0, 0] = 1
b[1, 0, 0, 1] = 1
b[0, 1, 1, 1] = 1
b[1, 0, 2, 2] = 1
b[0, 1, 0, 2] = 1

r[0, 0] = 2 ** (sites - 1)
r[1, 0] = 2 ** (sites - 1)

for site in range(sites):
    m = ax[site]

    rows = m[0].shape[0]
    columns = m[0].shape[1]

    n = np.zeros((d, rows * c, columns * c))

    for b1 in range(c):
        for b2 in range(c):
            for a1 in range(rows):
                for a2 in range(columns):
                    for sigma1 in range(d):
                        for sigma2 in range(d):
                            n[sigma1][b1 * rows + a1, b2 * columns + a2] += (b[sigma1, sigma2, b1, b2] *
                                                                             m[sigma2][a1, a2])

    a.append(n)

a[0] = l @ a[0]
a[-1] = a[-1] @ r

cx_check = convert.from_mps(a)
vx_check = np.zeros((Q, Q))

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

plt.imshow(vx)
plt.colorbar()
plt.title(r'$v_x$')

plt.xlabel('y')
plt.ylabel('x')

plt.show()

plt.imshow(vx_check)
plt.colorbar()
plt.title('MPO')

plt.xlabel('y')
plt.ylabel('x')

plt.show()
