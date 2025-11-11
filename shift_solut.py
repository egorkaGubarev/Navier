import json
import matplotlib.pyplot as plt
import numpy as np

import compress
import convert
import count
import shift
import utils

path = 'C:/Users/gubar/PycharmProjects/Navier'

Q = 256
K = 2
bound = 7
file_number = 4

d = 2 ** K
points = Q ** K
sites = int(np.log2(Q))
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

applied = utils.apply(convert.to_left_mps_with_svd(cx, sites, d, bound=bound), shift.left_2x(sites))
utils.left_with_qr(applied)
guess = compress.with_svd(applied, bound, unitary=False)
cx_check = convert.from_mps(guess)
vx_check = np.zeros((Q, Q))
mem = 0

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

for site in guess:
    mem += len(site.reshape(-1))

mem_save = 100 * mem / Q ** 2
classic = np.roll(vx, -2, 0)
differ = vx_check - classic
diff_norm = 100 * np.linalg.norm(differ) / np.linalg.norm(classic)
fig, ((left, right), (down_left, down_right)) = plt.subplots(2, 2)

origin = left.imshow(vx)
left.set_title(r'$v_x$')
fig.colorbar(origin)

left.set_xlabel('y')
left.set_ylabel('x')

applied = right.imshow(vx_check)
right.set_title('MPO')
fig.colorbar(applied)

right.set_xlabel('y')
right.set_ylabel('x')

im_classic = down_left.imshow(classic)
down_left.set_title('Class')
fig.colorbar(im_classic)

down_left.set_xlabel('y')
down_left.set_ylabel('x')

diff = down_right.imshow(differ)
down_right.set_title(fr'Diff, $\delta$ norm: {np.round(diff_norm, 1)} %, mem: {np.round(mem_save, 1)} %')
fig.colorbar(diff)

down_right.set_xlabel('y')
down_right.set_ylabel('x')

plt.tight_layout()
plt.show()