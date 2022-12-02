import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

eps = 0.00001

α = 0.002  # Вт/(см^2*град)
c = 1.65  # Дж/(cм^3*град)
l = 12  # см
T = 250  # с
k = 0.59  # Вт/(см*град)
R = 0.1


@njit(cache=True, nogil=True)
def φ_n(z) -> np.array:
    x1, x2 = 6, 8
    return 4 * 16 * (np.sin((x2 - l / 2) * 2 * z / l) - np.sin((x1 - l / 2) * 2 * z / l)) / (2 * z + np.sin(2 * z))


@njit(cache=True, nogil=True)
def half_method(n: int, a1: float, b1: float) -> np.ndarray:
    z = list()

    def find_c(a: float, b: float) -> float:
        G = α / (c * l)
        fz = lambda z: (np.tan(z) - G / z)
        root = (a + b) / 2
        while np.abs(a - b) > eps:
            if (fz(root) * fz(a)) < 0:
                b = root
            else:
                a = root
            root = (a + b) / 2
        return root

    while n > 0:
        root = find_c(a1, b1)
        z.append(root)
        a1 += np.pi
        b1 += np.pi
        n -= 1
    return np.array(z)


hx = 1 / 10
x_list = np.arange(0.0, l + hx, hx)
ht = 1 / 10
time_list = np.arange(0.0, T + ht, ht)


@njit(nogil=True, cache=True)
def w_n(z, φ, time=T, flag='x', x=l / 2):
    def P_n(zi, φi, t):
        a2 = (k / c) ** 2
        aRc = (2 * α / R) / (c ** 2)

        return (φi * 4 * (1 - np.exp(-1 * ((a2 * 4 * (zi ** 2) / (l ** 2) + aRc) * t))) / (
                a2 * 4 * (zi ** 2) / (l ** 2) + aRc)) / (c * l)

    def w(z, φ, ti, x):
        s = list()
        for i in prange(len(z)):
            s.append(P_n(z[i], φ[i], ti) * np.cos(((2 * z[i] / l) * (x - l / 2))))

        return np.sum(np.array(s))

    sol = []
    if flag == 'x':
        for i in x_list:
            sol.append(w(z, φ, time, i))
    elif flag == 't':
        for i in time_list:
            sol.append(w(z, φ, i, x))

    return sol


@njit(cache=True, nogil=True)
def solutions(n, t=T, flag='x', x=l / 2):
    def Rn(n):
        return 128 / ((c * l ** 2) * ((n + 1 / 2) ** 2) * np.pi ** 3) / α ** 2

    # while True:
    z = half_method(n, 0.000001, np.pi / 2)
    φ = φ_n(z)
    # df = pd.DataFrame({'z': z, 'φ': φ})
    # df.index = df.index + 1
    # print(df.to_string())
    # df.to_csv('values.csv')
    solution = w_n(z, φ, t, flag=flag, x=x)
    # if Rn(n + 1) / sum(solution) < eps:
    #     break
    # n += 1
    # print(t, n)
    return solution

def plotter(results_x, results_t):
    plt.figure(1)
    line = list()
    for key, value in results_x.items():
        line += plt.plot(x_list, value, label="t=" + str(key) + " с")
    plt.xlabel('Координата Х, См')
    plt.ylabel('Температура w, Кельвин')
    plt.legend(line, [l.get_label() for l in line], loc=0)
    plt.grid()

    plt.figure(2)
    line = list()
    for key, value in results_t.items():
        line += plt.plot(time_list, value, label="x=" + str(key) + " см")
    plt.xlabel('Время T, секунда')
    plt.ylabel('Температура w, Кельвин')
    plt.legend(line, [l.get_label() for l in line], loc=0)
    plt.grid()

    plt.show()


if __name__ == '__main__':
    t0 = time.time()
    n = 50
    t = 5
    numOfThreads = 8
    results_x = {}
    results_t = {}

    for i in prange(numOfThreads):
        results_x[(t + 35 * i)] = solutions(n + 35 * i, t + 35 * i)
    for i in prange(numOfThreads - 1):
        results_t[i] = solutions(n, flag='t', x=i)

    t1 = time.time()
    total = t1 - t0
    print(total)

    plotter(results_x, results_t)
