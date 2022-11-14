import numpy as np
import pandas as pd
import sympy
from sympy import symbols, integrate

eps = 0.00001

α = 0.002  # Вт/(см^2*град)
c = 1.65  # Дж/(cм^3*град)
l = 12  # см
T = 250  # с
k = 0.59  # Вт/(см*град)
R = 1  # ? Узанть коэф-т


def φ_n(z: list) -> np.array:
    return np.array(16 * np.sin(4 * np.array(z) / l) / (np.array(z) + np.sin(2 * np.array(z))))


def P_n(z, φ, t1=0, t_=1):
    t2 = T / 10
    t = symbols('t')
    return φ * sympy.exp(4 * (z ** 2) / c / (l ** 2) * (k + 2 * α / (c * R))) * integrate(sympy.exp(t), (t, t1, t2)) \
           - φ * integrate(t, (t, t1, t2)) * sympy.exp(-4 * (z ** 2) / c / (l ** 2) * (k + 2 * α / (c * R)) * t_)


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


if __name__ == '__main__':
    n = 10
    z = half_method(n, 0.000001, np.pi / 2)
    φ = φ_n(z)

    df = pd.DataFrame({'z': z, 'φ': φ})
    df.index = df.index + 1
    print(df.to_string())
    df.to_csv('values.csv')
    print()
    print(P_n(z, φ))
