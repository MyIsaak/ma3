import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except:
    print("Import required modules")
    sys.exit(1)


def funktion(x):
    y = np.sin(x)
    return y


def ableitung(f, x):
    h = 0.00001
    N = len(x)
    f_strich = np.zeros(N)

    for i in range(0, N):
        f_strich[i] = (f(x[i] + h) - f(x[i] - h)) / (2*h)

    return f_strich


x_end = 4*np.pi
tau = 0.1
x = np.arange(0, x_end, tau)

f = funktion(x)
f_f_strich = ableitung(funktion, x)

plt.plot(x, f, x, f_f_strich, linewidth='1')
plt.grid(True)
plt.legend(['Funktion', 'Ableitungsfunktion'])
