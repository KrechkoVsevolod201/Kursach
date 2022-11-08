import matplotlib.pyplot as plt
import numpy as np


def half_method(name):
    n = 100
    a1 = 0.00001
    b1 = np.pi/2
    a = a1
    b = b1
    i = 0
    eps = 0.00001
    s = []
    fz = lambda z: (np.tan(z) - 1/z)

    while n > 0:
        a = a1
        b = b1
        while np.abs(a - b) > eps:
            c = (a + b) / 2
            if (fz(c) * fz(a)) < 0:
                b = c
            else:
                a = c
            #i = i + 1
        s.insert(100 - n, c)
        a1 = a1 + np.pi
        b1 = b1 + np.pi
        n = n - 1
        print(c)
#    print(s)  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    half_method('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
