import math
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange, float64, int64


# eps = 0.00001
# α = 0.002  # Вт/(см^2*град)
# c = 1.65  # Дж/(cм^3*град)
# l = 12  # см
# T = 250  # с
# k = 0.59  # Вт/(см*град)
# R = 0.1

@njit(nogil=True)
def φ_n(z, l) -> np.array:
    x1, x2 = 6, 8
    return 4 * 16 * (np.sin((x2 - l / 2) * 2 * z / l) - np.sin((x1 - l / 2) * 2 * z / l)) / (2 * z + np.sin(2 * z))


@njit(nogil=True)
def half_method(n: int, a1: float, b1: float, eps, l, c, α) -> np.ndarray:
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


@njit(nogil=True)
def w_n(z, φ, time, c, k, R, l, α, x_list, x, time_list, flag='x'):
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


@njit(nogil=True)
def solutions(n, t, α, c, l, k, R, eps, x_list, time_list, x, flag='x'):
    z = half_method(n, 0.000001, np.pi / 2, eps, l, c, α)
    φ = φ_n(z, l)

    solution = w_n(z=z, φ=φ, α=α, c=c, l=l, k=k, R=R, time=t, flag=flag, x=x, x_list=x_list, time_list=time_list)

    return solution, z, φ


# @njit(nogil=True, cache=True)
# def w_l2T(z, φ, time, x):
#     def P_n(zi, φi, t):
#         a2 = (k / c) ** 2
#         aRc = (2 * α / R) / (c ** 2)
#
#         return (φi * 4 * (1 - np.exp(-1 * ((a2 * 4 * (zi ** 2) / (l ** 2) + aRc) * t))) / (
#                 a2 * 4 * (zi ** 2) / (l ** 2) + aRc)) / (c * l)
#
#     def w(z, φ, ti, x):
#         s = list()
#         for i in prange(len(z)):
#             s.append(P_n(z[i], φ[i], ti) * np.cos(((2 * z[i] / l) * (x - l / 2))))
#
#         return np.sum(np.array(s))
#
#     sol = w(z, φ, time, x)
#
#     return sol


@njit(float64(float64, int64))
def truncate(f, accuracy):
    return math.floor(f * 10 ** accuracy) / 10 ** accuracy


# @njit(fastmath=True, nogil=True, cache=True)
# def solutionN(n, epsilon=10 ** (-2), accuracy=2):
#     def Rn(n):
#         return 32 / ((c * l) * ((n) ** 2) * np.pi ** 3) / α ** 2
#
#     while True:
#         z = half_method(n, 0.000001, np.pi / 2)
#         φ = φ_n(z)
#
#         solution_with_n = w_l2T(z, φ)
#         # print("Rn: ", Rn(n + 1))
#         if Rn(n + 1) <= epsilon:
#             break
#         n += 1
#
#     solution_with_n = truncate(solution_with_n, accuracy)
#     N: int64
#     N = n
#     while True:
#         N -= 1
#         z = half_method(N, 0.000001, np.pi / 2)
#         φ = φ_n(z)
#         solution_with_N = truncate(w_l2T(z, φ), accuracy)
#
#         if np.abs(solution_with_n - solution_with_N) > epsilon:
#             N += 1
#             break
#     return [n, N]


# def plotter(results_x, results_t):
#     plt.figure(1)
#     line = list()
#     for key, value in results_x.items():
#         line += plt.plot(x_list, value, label="t=" + str(key) + " с")
#     plt.xlabel('Координата Х, cм')
#     plt.ylabel('Температура w, К')
#     plt.legend(line, [l.get_label() for l in line], loc=0)
#     plt.grid()
#
#     plt.figure(2)
#     line = list()
#     for key, value in results_t.items():
#         line += plt.plot(time_list, value, label="x=" + str(key) + " см")
#     plt.xlabel('Время T, с')
#     plt.ylabel('Температура w, К')
#     plt.legend(line, [l.get_label() for l in line], loc=0)
#     plt.grid()
#
#     plt.show()


def m(n, α, c, l, T, k, R, eps):
    hx = 1 / 10
    x_list = np.arange(0.0, l + hx, hx)
    ht = 1 / 10
    time_list = np.arange(0.0, T + ht, ht)

    t = 5
    numOfThreads = 8
    results_x = {}
    results_t = {}

    for i in prange(numOfThreads):
        [results_x[(t + 35 * i)], z1, φ1] = solutions(n=n, t=t + 35 * i, α=α, c=c, l=l, k=k, R=R, eps=eps,
                                                      x=l / 2,
                                                      flag='x',
                                                      x_list=x_list, time_list=time_list)
    for i in prange(numOfThreads - 1):
        [results_t[i], z1, φ1] = solutions(n=n, t=T, α=α, c=c, l=l, k=k, R=R, eps=eps, flag='t', x=i, x_list=x_list,
                                           time_list=time_list)
    # plotter(results_x, results_t)
    # N = np.array([1140, 3600, 11410, 36090, 114150, 360980, 1141520])
    # for i in prange(2, 9):
    #     n = N[i - 2]
    #     s1, s2 = solutionN(n, epsilon=10 ** (-i), accuracy=i)
    #     print(f"Для epsilon: 10**(-{i})", " n=", s1, "N=", s2)

    return results_x, x_list, results_t, time_list, z1, φ1
