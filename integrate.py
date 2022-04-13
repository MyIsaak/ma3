import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except:
    print("Import required modules")
    sys.exit(1)


def funktion(x):
    return np.cos(x)


def integral(f, xmin, xmax):
    h = 0.01
    N = int((xmax - xmin) / h)
    I = 0
    x = xmin

    # 1,2, ..., N aber range(0, N) -> 0, 1, 2, ..., N
    for i in range(N):
        I = I + (f(x) + f(x + h)) / 2*h
        x = x+h
    return I


# Berechnung der Integralfunktion
x_0 = 0
x_end = 4*np.pi
dx = 0.1
x = np.arange(0, x_end, dx)
N = len(x)
f_integral = np.zeros(N)

for i in range(0, N):
    f_integral[i] = integral(funktion, x_0, x[i])

f = funktion(x)

plt.plot(x, f, x, f_integral, linewidth='1')
plt.grid(True)
plt.legend(['Funktion', 'Integralfunktion'], loc='upper right')
