import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except:
    print("Import matplotlib and numpy")
    sys.exit(1)


def f(x):
    return 3*x**2 - 1 - np.exp(x)  # work with arrays of x


def f_strich(x, h):
    return (f(x+h) - f(x)) / h


def newton(f, x_start):
    h = 0.001
    x = x_start
    e = 0.00001

    for i in range(99):
        x_neu = x-f(x)/f_strich(x, h)
        if abs(x_neu - x) < e:
            return x_neu
        x = x_neu


def plot_nullstellen(x_min, x_max, dx, f):
    x = np.arange(x_min, x_max, dx)
    y = f(x)
    n_max = int((x_max - x_min)/dx)
    ii = 0
    x_zero = []

    for i in range(n_max - 1):
        if y[i] * y[i+1] <= 0:
            x_start = (x[i]+x[i+1]) / 2
            x_zero.append(newton(f, x_start))
            ii = ii + 1

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(x, y)
    ax1.set_xlabel('x-Achse')
    ax1.set_ylabel('y_Achse')
    ax1.set_title('Funktion, deren Nullstellen berechnet werden')
    ax1.grid(True)
    txt = 'Nullstellen: '
    print(x_zero)
    for i in range(ii):
        if x_zero[i] != None:
            txt = txt + '{:4.2f}     '.format(x_zero[i])

    ax2.set_axis_off()
    ax2.text(0.0, 0.5, txt)
    fig.savefig('Nullstellensuche.pdf')


# plot_nullstellen(-1.5, 1.5, 0.1, f)
