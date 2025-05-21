# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

# Main function
def main():
    # Load data
    file_path = os.path.expanduser('~/Documents/GitHub/FHLN40/Project/SINDy_input_data.csv')
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    t, x1, x2 = data[:,0], data[:,1], data[:,2]
    X = np.vstack([x1, x2]).T

    # Central difference derivative
    dt = t[1] - t[0]
    X_dot = np.empty_like(X)
    X_dot[1:-1] = (X[2:] - X[:-2])/(2*dt)
    X_dot[0] = (X[1] - X[0])/dt
    X_dot[-1] = (X[-1] - X[-2])/dt

    # Plot the raw data
    plt.figure(figsize=(12,8))
    plt.plot(t,x1,'-',label=r'$x_1$')
    plt.plot(t,x2,'--',label=r'$x_2$')
    plt.xlabel('t'); plt.legend(); plt.title('Raw time series')
    plt.tight_layout()
    plt.show()

    # Create SINDy models
    models = {}
    models['Polynomial of the second degree'] = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2), optimizer=ps.STLSQ(threshold=0.01)) # Second-order polynomial 
    models['Polynomial of the third degree'] = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3), optimizer=ps.STLSQ(threshold=0.01)) # Third-order polynomial 

    # Fit each model
    for name, m in models.items():
        m.fit(X, x_dot=X_dot, t=dt)
        print(f"Identified equation: {name}")
        m.print()

    # Simulate and compare
    t_sim = np.linspace(0,2,200)
    x0 = X[0]

    # Plot the comparison
    plt.figure(figsize=(12,8))
    for i, (name, m) in enumerate(models.items(),1):
        X_sim = m.simulate(x0, t_sim)
        plt.subplot(1, len(models), i)
        plt.plot(t, x1, 'k', label='Data $x_1$')
        plt.plot(t, x2, 'k--', label='Data $x_2$')
        plt.plot(t_sim, X_sim[:,0], 'r', label='Simulated $x_1$')
        plt.plot(t_sim, X_sim[:,1], 'r--', label='Simulated $x_2$')
        plt.title(name)
        plt.xlabel('t') 
        plt.legend() 
        plt.tight_layout()

    plt.show()

# Run the main function
if __name__ == "__main__":
    main()