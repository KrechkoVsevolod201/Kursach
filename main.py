import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from sympy import N

eps = 0.00001

α = 0.002  # Вт/(см^2*град)
c = 1.65  # Дж/(cм^3*град)
l = 12  # см
T = 250  # с
k = 0.59  # Вт/(см*град)
R = 0.1  # ? Узанть коэф-т


def φ_n(z: list) -> np.array:
    z = np.array(z)
    x = sympy.Symbol('x')
    lis = list()

    for zi in z:
        lis.append(
            (4 * sympy.integrate(sympy.cos((2 * zi / l) * x), (x, 6, 8))) / (
                    2 * zi + 2 * sympy.sin(2 * zi)))
    return np.array(lis)
    # return np.array(2 * 16 * np.sin(4 * np.array(z) / l) / (np.array(z) + np.sin(2 * np.array(z))))


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


a2 = (k / c) ** 2
aRc = (2 * α / R) / (c ** 2)
q = lambda zi: a2 * 4 * (zi ** 2) / (l ** 2)


def w_n(z, φ, time=1, th=1 / 100, x=0):
    def P_n(zi, φi, t):
        return (φi * 4 * (1 - sympy.exp(-1 * ((q(zi) + aRc) * t))) / (q(zi) + aRc)) / (c * l)

    def w(z, φ, ti, x):
        s = list()

        for zi, φi in zip(z, φ):
            s.append(N(P_n(zi, φi, ti) * sympy.cos(((2 * zi / l) * (x - l / 2)))))
        return sum(s)

    t_list = np.arange(0.0, time + th, th)

    s = list(map(lambda ti: w(z, φ, ti, x), t_list))

    return [s, t_list]


def plotter(wi, ti):
    plt.plot(ti, wi)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    n = 10
    z = half_method(n, 0.000001, np.pi / 2)
    φ = φ_n(z)
    df = pd.DataFrame({'z': z, 'φ': φ})
    df.index = df.index + 1
    print(df.to_string())
    df.to_csv('values.csv')

    [solution, t_i] = w_n(z, φ, 250, 1 / 10, 3)
    plotter(solution, t_i)
