import sys

from numpy import arange

try:
    import matplotlib.pyplot as plt
    import numpy as np
except:
    print("Import matplotlib and numpy")
    sys.exit(1)

omega = 0.5*np.pi


def dgl(x, t):
    dxdt = np.sin(omega*t) - x
    return dxdt


def euler(f, x_n, t_n, tau):
    x_n1 = x_n+tau*(f(x_n, t_n))
    return x_n1


x0 = 0.5
x = []
x.append(x0)
t_end = 20
tau = 0.01
t = np.arange(0, t_end, tau)

N = int(t_end/tau)

for i in range(1, N):
    x.append(euler(dgl, x[i-1], t[i-1], tau))

plt.plot(t, x, t, np.sin(omega*t))
plt.xlabel('t')
plt.ylabel('x')
plt.grid(True)
plt.legend(['Lösung der DGL', 'äußere Anregung der sinsus'])
plt.title('Lösung der DGL dx/dt=sin(omega*t)-x, x(0)=0.5')
print('end')

plt.savefig('Nullstellensuche.pdf')
