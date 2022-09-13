import pandas as pd
from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# 直线方程函数
def f_1(x, A, B):
    return A * x + B


# 二次曲线方程
def f_2(x, A, B, C):
    return A * x * x + B * x + C


# 三次曲线方程
def f_3(x, A, B, C, D):
    return A * x * x * x + B * x * x + C * x + D




def Gaussian(x, A, miu, sigma):
    return A * np.exp(-(x - miu) ** 2 / (2 * sigma ** 2))


def plot_test(x0, y0):
    plt.figure()

    plt.scatter(x0[:], y0[:], 25, "red")

    A, miu, sigma = optimize.curve_fit(Gaussian, x0, y0,bounds=(0, [6,13,20,120,4,6]))[0]
    print(A, miu, sigma)
    x1 = x0
    y1 = A * np.exp(-(x1 - miu) ** 2 / 2 / sigma ** 2)
    plt.plot(x1, y1, "blue")

    plt.title("test")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

    return


def fitting_1(x, y):
    g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, x, y)
    print(g)


if __name__ == '__main__':
    names = ['wavelength', 'L', 'M', 'S']
    data = pd.read_csv(r"D:\IDM\linss10e_fine.csv", names=names)
    wavelength = np.array(data["wavelength"]).tolist()
    L = np.array(data["L"]).tolist()
    M = np.array(data["M"]).tolist()
    S = np.array(data["S"]).tolist()
    LMS_turple_list = []
    for i in range(len(wavelength)):
        LMS_turple_list.append((L[i], M[i], S[i]))
    dict_LMS = dict(zip(wavelength, LMS_turple_list))
    x = np.linspace(400, 700, 31)
    # print(x)
    L_array = []
    M_array = []
    S_array = []
    for i in x:
        l, m, s = dict_LMS[i]
        L_array.append(l)
        M_array.append(m)
        S_array.append(s)
    L_array = np.array(L_array)
    M_array = np.array(M_array)
    S_array = np.array(S_array)

    plt.plot(x, L_array)
    plt.plot(x, M_array)
    plt.plot(x, S_array)
    plt.show()
    # fitting_1(x, L_array)
    plot_test(x, L_array)
