import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def der(data, axis):
    return (np.roll(data, -1, axis) - np.roll(data, 1, axis)) / (2 * step)

def der_2(data, axis):
    return (np.roll(data, -1, axis) - 2 * data + np.roll(data, 1, axis)) / step ** 2

def solve_poisson_2d_fft(n, f):
    kx = 2 * np.pi * np.fft.fftfreq(n, d=1 / n)
    ky = 2 * np.pi * np.fft.fftfreq(n, d=1 / n)
    KX, KY = np.meshgrid(kx, ky)

    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1

    F_hat = np.fft.fft2(f)

    U_hat = -F_hat / K2
    U_hat[0, 0] = 0

    return np.real(np.fft.ifft2(U_hat))

path = 'C:/Users/gubar/PycharmProjects/Navier'

n = 256
h = 1 / 200
dim = 2
tend = 2
Re = 1000

y_min = 0.4
y_max = 0.6

dt = 0.001
plots = 5
w_thresh = 6

step = 1 / (n - 1)
time_steps = int(tend / dt)

J = np.zeros((dim, n, n))
D = np.zeros((dim, n, n))
ones = np.ones(n)

for j in range(n):
    y = j * step

    arg_min = (y - y_min) / h
    arg_max = (y - y_max) / h

    J[0, :, j] = ones * (np.tanh(arg_min) - np.tanh(arg_max) - 1) / 2

    for i in range(n):
        x = i * step

        D[0, i, j] = (2 * (arg_max * np.exp(- arg_max ** 2) + arg_min * np.exp(- arg_min ** 2)) *
                      (np.sin(8 * np.pi * x) + np.sin(24 * np.pi * x) + np.sin(6 * np.pi * x)) / h)
        D[1, i, j] = (np.pi * (np.exp(- arg_max ** 2) + np.exp(- arg_min ** 2)) *
                      (8 * np.cos(8 * np.pi * x) + 24 * np.cos(24 * np.pi * x) + 6 * np.cos(6 * np.pi * x)))

D /= (np.max(np.sqrt(D[0] ** 2 + D[1] ** 2)) / 8)

vx = J[0] + D[0]
vy = J[1] + D[1]

vx -= np.mean(vx)
vy -= np.mean(vx)

w_0 = der(vy, 0) - der(vx, 1)
w_hist = [w_0]
psi_hist = [solve_poisson_2d_fft(n, -w_0)]

for time_step in tqdm.tqdm(range(time_steps)):
    w = w_hist[-1]
    psi = psi_hist[-1]

    w_hist.append(w + dt * (der(psi, 0) * der(w, 1) - der(psi, 1) * der(w, 0) + (der_2(w, 0) + der_2(w, 1)) / Re))
    psi_hist.append(solve_poisson_2d_fft(n, -w_hist[-1]))

fig, ax = plt.subplots(1, plots)

for plot in range(plots):
    progress = plot / (plots - 1)
    it = int(time_steps * progress)
    time = tend * progress

    w = w_hist[it]
    psi = psi_hist[it]

    im = ax[plot].imshow(np.where(np.abs(w) > w_thresh, w, 0), cmap='bwr')
    ax[plot].set_title(f't = {np.round(time, 2)}')

    ax[plot].set_xlabel('y')
    ax[plot].set_ylabel('x')

    with open(f'{path}/data/result_{plot}.json', 'w') as file:
        vx = []
        vy = []

        for string in der(psi, 1):
            vx.append(list(string))

        for string in -der(psi, 0):
            vy.append(list(string))

        json.dump({'time': time, 'vx': vx, 'vy': vy}, file)

fig.colorbar(im, ax=ax.ravel().tolist(), orientation='horizontal')
plt.show()
