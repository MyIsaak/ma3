import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except:
    print("Import required modules")
    sys.exit(1)

from differentiate import ableitung
from nullstellen import plot_nullstellen

# Aufgabe 55


def funktionA(u):
    return u * (1.5 - u**2) + 1


def funktionB(x):
    return -np.log(np.sqrt(x)) + 4*np.exp(-0.3 * x)


def funktionC(x):
    return x + np.exp(x) * 0.5


def funktionD(x):  # für x = [-1, 0]
    return np.exp(x) - x**2


plot_nullstellen(-10, 10, 1e-5, funktionA)
plot_nullstellen(1, 10, 1e-4, funktionB)
plot_nullstellen(-10, 10, 1e-4, funktionC)
plot_nullstellen(-1, 0, 1e-4, funktionD)
