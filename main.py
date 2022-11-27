import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from sympy import N
from multiprocessing.pool import ThreadPool

eps = 0.00001

α = 0.002  # Вт/(см^2*град)
c = 1.65  # Дж/(cм^3*град)
l = 12  # см
T = 250  # с
k = 0.59  # Вт/(см*град)
R = 1


def φ_n(z: list) -> np.array:
    z = np.array(z)

    x1, x2 = 6, 8
    return np.array(
        4 * 16 * (np.sin((x2 - l / 2) * 2 * z / l) - np.sin((x1 - l / 2) * 2 * z / l)) / (2 * z + np.sin(2 * z)))


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


def w_n(z, φ, time=1):
    def P_n(zi, φi, t):
        a2 = (k / c) ** 2
        aRc = (2 * α / R) / (c ** 2)
        q = lambda zi: a2 * 4 * (zi ** 2) / (l ** 2)

        return (φi * 4 * (1 - sympy.exp(-1 * ((q(zi) + aRc) * t))) / (q(zi) + aRc)) / (c * l)

    def w(z, φ, ti, x):
        s = list()

        for zi, φi in zip(z, φ):
            s.append(N(P_n(zi, φi, ti) * sympy.cos(((2 * zi / l) * (x - l / 2)))))
        return sum(s)

    s = list(map(lambda xi: w(z, φ, time, xi), x_list))

    return s


def plotter(results):
    fig, ax = plt.subplots()
    for i in results:
        ax.plot(x_list, i)
    ax.grid()

    #  Добавляем подписи к осям:
    ax.set_xlabel('Координата Х, См')
    ax.set_ylabel('Температура w, Кельвин')
    plt.show()


def solutions(n, t):
    Rn = lambda n: 128 / ((c * l ** 2) * ((n + 1 / 2) ** 2) * sympy.pi ** 3) / α ** 2
    # while True:
    z = half_method(n, 0.000001, np.pi / 2)
    φ = φ_n(z)
    df = pd.DataFrame({'z': z, 'φ': φ})
    df.index = df.index + 1
    # print(df.to_string())
    df.to_csv('values.csv')
    solution = w_n(z, φ, t)
    # if N(Rn(n + 1)) / sum(solution) < eps:
    #     break
    # n += 1
    return solution


if __name__ == '__main__':
    numOfThreads = 5
    results = []
    pool = ThreadPool(numOfThreads)
    results.append(pool.apply_async(solutions, (300, 350)))
    results.append(pool.apply_async(solutions, (238, 250)))
    results.append(pool.apply_async(solutions, (120, 200)))
    results.append(pool.apply_async(solutions, (50, 150)))
    results.append(pool.apply_async(solutions, (30, 100)))

    results = [r.get() for r in results]

    pool.close()
    pool.join()

    print(results)
    plotter(results)
