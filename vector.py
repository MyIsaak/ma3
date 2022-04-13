import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.linalg as npl
    from scipy.fft import fft, ifft
except:
    print("Import required modules")
    sys.exit(1)

a = np.array([0, 3, 4])
b = np.array([2, 3, 1])

print('Skalarprodukt 1 =', a.dot(b))
print('Skalarprodukt 1 =', np.dot(a, b))
print('Skalarprodukt 1 =', a @ b)

print('Betrag von a =', np.sqrt(a.dot(a)))

A = np.array([[1, 2], [2, 0], [3, 1]])  # 3x2 Matrix
B = np.array([[1, 2, 3], [2, 1, 2]])  # 3x2 Matrix

# Matrix mul ist nur scalar producten von spalten und zeilen
print('A mal B = \n', A.dot(B))
print('B mal A = \n', B.dot(A))

# Problem! keine Unterschied zw. spalten und zeilen vektoren

C = np.array([[2, 1, 0], [2, 3, 1]])
print('B * C =\n', B*C)  # Achtung! Keine mat. mul, sondern elementare mul

print('a x b =', np.cross(a, b))

A = np.array([[1, 3], [2, 4]])
print('det(A) =', npl.det(A))

# Eigenwerte und Eigenvektoren
A = np.array([[1, -3, 3], [3, -5, 3], [6, -6, 4]])
EW, EV = npl.eig(A)  # EW = (lam1, lam2, lam3)
txt = 'Eigenwerte und Eigenvektoren :\n'
laenge = len(EW)
print(laenge)

# Wir wollen nur abs
'''
for k in range(laenge):
    if abs(EW[k].imag) < 10**-7:
        print('Eigenwert =', round(EW[k].real, 4))
        print('Eigenvektor', EV[:, k])

A = np.array([[1, -1, 2], [-1, 2, 1], [0, 0, 1]]
print('Matrix A = \n', A)

# Achtung! Python invertieren matrixen obwohl det=0 sein kann
if abs(npl.det(A)) > 10**-5:
    A_inv=npl.inv(A)
    print('inverse Matrix von A =\n', A_inv)
else:
    print('singulare Matrix - inverse Matrix existiert nicht')

# Losung linearer Gleichungssysteme
b=np.array([1, 4, 2])

if abs(npl.det(A)) > 10**-5:
    x=npl.solve(A, b)  # A*x = b
    print('Lösung von Ax=b; x =\n', x)
else:
    print('singulare Matrix - keine eindeutige Lösung')
'''

# Zeit signal erzeugen
N = 1024
T = 100
dt = T/N
t = np.arange(0, 100, dt)
x = np.zeros(N)
fi = np.array([30/T, 100/T, 300/T])  # M = 3
Ai = np.array([10, 3, 5])

for k in range(0, len(Ai)):
    x += Ai[k] * np.sin(2*np.pi*fi[k]*t)

plt.plot(t, x)

# FFT durchführen
X = fft(x)

# Amplitude des Spektrums berechnen
P = np.abs(X) / (N/2)  # normierte werte
# zugehorige Freqzuenzen berechnen:
frq = np.arange(0, N/2, 1)
fig, ax = plt.subplots(2, 1, figsize=(8.3, 11.7))
ax[0].set_title('zeitabhängiges Signal')
ax[0].plot(t, x, linewidth=1)
ax[0].set_ylabel('Signalstärke in a.u.')
ax[0].set_xlabel('Zeit in s')

ax[1].set_title('Amplitudenspektrum')
ax[1].plot(frq, P[0:int(N/2)], linewidth=1)
ax[1].set_ylabel('Amplitude in a.u.')
ax[1].set_xlabel('Frequenz in Hz')
