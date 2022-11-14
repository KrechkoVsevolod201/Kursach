import csv
import pandas as pd
import numpy as np

eps = 0.00001


def φ_n(μ: list) -> np.ndarray:
    return 4 * 16 * np.sin(2 * np.array(μ)) / (12 * np.array(μ) + 2 * np.sin(np.array(μ) * 12))


def half_method(n: int, a1: float, b1: float) -> list:
    μ = list()

    def find_c(a: float, b: float) -> float:
        G = 1
        fz = lambda z: (np.tan(z) - G / z)
        c = (a + b) / 2
        while np.abs(a - b) > eps:
            if (fz(c) * fz(a)) < 0:
                b = c
            else:
                a = c
            c = (a + b) / 2
        return c

    while n > 0:
        c = find_c(a1, b1)
        μ.append(c)
        a1 += np.pi
        b1 += np.pi
        n -= 1
    return μ


if __name__ == '__main__':
    μ = half_method(100, 0.000001, np.pi / 2)
    φ = φ_n(μ)

    df = pd.DataFrame({'μ': μ, 'φ': φ})
    print(df.to_string())
    df.to_csv('values.csv')
