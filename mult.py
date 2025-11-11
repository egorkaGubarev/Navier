import json
import matplotlib.pyplot as plt
import numpy as np

import compress
import convert
import count
import utils

path = 'C:/Users/gubar/PycharmProjects/Navier'

K = 2
bound = 11
file_number = 4

shrink = 0.2

with open(f'{path}/data/result_{file_number}.json') as file:
    result = json.load(file)

vx = np.array(result['vx'])
vy = np.array(result['vy'])

Q = vx.shape[0]
d = 2 ** K
points = Q ** K
sites = int(np.log2(Q))
h = 1 / Q
classic = vx * vy
check_2d = np.zeros((Q, Q))

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

mps_x = convert.to_left_mps_with_svd(cx, sites, d, bound)
mps_y = convert.to_left_mps_with_svd(cy, sites, d, bound)
prod = utils.mult(mps_x, mps_y)
utils.left_with_qr(prod)
guess = compress.with_svd(prod, bound, False)
check = convert.from_mps(guess)
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

    check_2d[int(x), int(y)] = check[point]

for site in guess:
    mem += len(site.reshape(-1))

differ = check_2d - classic
diff_norm = 100 * np.linalg.norm(differ) / np.linalg.norm(classic)
mem_save = 100 * mem / Q ** 2

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

im_mps = ax1.imshow(check_2d)
ax1.set_title('MPS')
fig.colorbar(im_mps, shrink=shrink)

ax1.set_xlabel('y')
ax1.set_ylabel('x')

im_classic = ax2.imshow(classic)
ax2.set_title('Class')
fig.colorbar(im_classic, shrink=shrink)

ax2.set_xlabel('y')
ax2.set_ylabel('x')

im_diff = ax3.imshow(differ)
ax3.set_title(fr'Diff, $\delta$ norm: {np.round(diff_norm, 1)} %, mem: {np.round(mem_save, 1)} %')
fig.colorbar(im_diff, shrink=shrink)

ax3.set_xlabel('y')
ax3.set_ylabel('x')

plt.tight_layout()
plt.show()
