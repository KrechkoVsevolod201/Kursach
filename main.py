import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter
# Если автоимпорт не работает, то надо прописать в терминале "pip install XlsxWriter"


def half_method():
    n = 100
    a1 = 0.00001
    b1 = np.pi/2
    eps = 0.00001
    root = []
    fz = lambda z: (np.tan(z) - 1/z)
    # открываем новый файл на запись
    workbook = xlsxwriter.Workbook('z.xlsx')
    # создаем там "лист"
    worksheet = workbook.add_worksheet()
    # в ячейку A1 пишем текст
    worksheet.write('A1', 'Корни z')
    while n > 0:
        a = a1
        b = b1
        while np.abs(a - b) > eps:
            c = (a + b) / 2
            if (fz(c) * fz(a)) < 0:
                b = c
            else:
                a = c
        root.insert(100 - n, c)
        # в ячейку A пишем текст
        worksheet.write('A' + str(100-n+2), c)
        a1 = a1 + np.pi
        b1 = b1 + np.pi
        n = n - 1
    print(root)

    # сохраняем и закрываем
    workbook.close()


def plotter():
    fz1 = lambda z: (np.tan(z))
    fz2 = lambda z: (-1/z)
    z = np.linspace(-10, 10, 100000)
    plt.plot(z, fz1(z))
    plt.plot(z, fz2(z))
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    half_method()
