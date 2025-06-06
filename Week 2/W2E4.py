import numpy as np
import matplotlib.pyplot as plt

def genereate_tan_data(m):
    x = np.linspace(-1.0, 1.0, m)
    y = np.tan(x)
    noise = np.random.normal(0, 0.1, size=y.shape)
    y += noise
    return x, y

def plot_data(x,y):
    plt.scatter(x,y, color='green')
    plt.show()

def design_matrix(x):
    X = np.vstack((np.ones(np.size(x)), x)).T
    return X

def solve_linear_regression(X, y):
    LSS = np.linalg.solve(X.T @ X, X.T @ y)
    return LSS

def plot_regression_line(x,y, LSS):
    plt.scatter(x, y, label='Data points', color='blue')
    plt.plot(x, LSS[0] + LSS[1] * x, label='Regression Line', color='red')
    plt.legend()
    plt.show()
    
def main():
    x, y = genereate_tan_data(10)
    plot_data(x,y)
    X = design_matrix(x)
    LSS = solve_linear_regression(X, y)
    plot_regression_line(x,y, LSS)

main()