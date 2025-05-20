# Fix kvar

# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

# Load data
file_path = os.path.expanduser('~/Documents/GitHub/FHLN40/Project/SINDy_input_data.csv')
data = np.loadtxt(file_path, delimiter=',', skiprows=1)
t, x1, x2 = data[:,0], data[:,1], data[:,2]
X = np.vstack([x1, x2]).T

# Central difference derivative
dt = t[1] - t[0]

# PySINDy has its own differentiation, but here’s a simple FD:
X_dot = np.empty_like(X)
X_dot[1:-1] = (X[2:] - X[:-2])/(2*dt)
X_dot[0] = (X[1] - X[0])/dt
X_dot[-1] = (X[-1] - X[-2])/dt

# Plot the data
plt.figure(figsize=(8,4))
plt.plot(t,x1,'-',label=r'$x_1$')
plt.plot(t,x2,'--',label=r'$x_2$')
plt.xlabel('t'); plt.legend(); plt.title('Raw time series')
plt.tight_layout()
plt.show()

# Create SINDy models
models = {}

# (a) 2nd-order polynomial library
poly2 = ps.PolynomialLibrary(degree=2)
optimizer = ps.STLSQ(threshold=0.01)  # sequential thresholded least-squares
models['poly2'] = ps.SINDy(feature_library=poly2, optimizer=optimizer)

# (b) 3rd-order polynomial + Fourier library
poly3 = ps.PolynomialLibrary(degree=3)
fourier = ps.FourierLibrary(n_frequencies=2)  # includes sin(ωx), cos(ωx) up to 2ω
large_lib = poly3 + fourier
models['poly3+fourier'] = ps.SINDy(feature_library=large_lib,
                                   optimizer=ps.STLSQ(threshold=0.01))

# 4) Fit both models
for name, m in models.items():
    # pass your own derivative estimates in x_dot
    m.fit(X, x_dot=X_dot, t=dt)
    print(f"--- Identified equations ({name}) ---")
    m.print()

# 5) Simulate forward on [0,2] and compare
t_sim = np.linspace(0,2,200)
x0 = X[0]

plt.figure(figsize=(10,4))
for i, (name, m) in enumerate(models.items(),1):
    X_sim = m.simulate(x0, t_sim)
    plt.subplot(1,2,i)
    plt.plot(t, x1, 'k', label='data $x_1$')
    plt.plot(t, x2, 'k--', label='data $x_2$')
    plt.plot(t_sim, X_sim[:,0], 'r', label='sim $x_1$')
    plt.plot(t_sim, X_sim[:,1], 'r--', label='sim $x_2$')
    plt.title(name)
    plt.xlabel('t'); plt.legend(); plt.tight_layout()

plt.show()

