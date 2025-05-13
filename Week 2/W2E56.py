import numpy as np
import matplotlib.pyplot as plt
 
def poly_regression(x_train, y_train, p):
   
    X = np.vstack([x_train**i for i in range(p+1)]).T 
    w = np.linalg.pinv(X.T @ X) @ X.T @ y_train

    return w

def polynomial_predict(x, w):
    
    p = len(w) - 1 
    X = np.vstack([x**i for i in range(p+1)]).T 

    return X @ w

def plot_poly_fit(x_train, y_train, x_validation, y_validation, x_test, y_test, w, p):
   
    plt.figure()
    
    plt.scatter(x_train, y_train, color='blue', label='Training set')
    plt.scatter(x_validation, y_validation, color='orange', label='Validation set')
    plt.scatter(x_test, y_test, color='gray', label='Test set')

    x_grid = np.linspace(np.min(x_train), np.max(x_train), 200)
    y_grid = polynomial_predict(x_grid, w)
    plt.plot(x_grid, y_grid, 'r-', label=f'Polynomial p={p}')

    plt.title(f'Polynomial Regression (p={p})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def mean_squared_error(x, y, w):
    
    y_pred = polynomial_predict(x, w)

    return np.mean((y - y_pred)**2)

def main():
    x_train = np.loadtxt('./week02_input_students_part2/x_train.txt')
    y_train = np.loadtxt('./week02_input_students_part2/y_train.txt')
    x_validation = np.loadtxt('./week02_input_students_part2/x_validation.txt')
    y_validation = np.loadtxt('./week02_input_students_part2/y_validation.txt')
    x_test = np.loadtxt('./week02_input_students_part2/x_test.txt')
    y_test = np.loadtxt('./week02_input_students_part2/y_test.txt')

    # EXERCISE 5:
    for p in [1, 4, 11]:
        w = poly_regression(x_train, y_train, p)
        plot_poly_fit(x_train, y_train,
                      x_validation, y_validation,
                      x_test, y_test, w, p)

    # EXERCISE 6:
    degrees = range(1, 12)
    mse_train = []
    mse_val   = []
    mse_test  = []
    for p in degrees:
        w = poly_regression(x_train, y_train, p)
        mse_train.append(mean_squared_error(x_train, y_train, w))
        mse_val.append(mean_squared_error(x_validation, y_validation, w))
        mse_test.append(mean_squared_error(x_test, y_test, w))

    plt.figure()
    plt.plot(degrees, mse_train, 'o-', label='Train MSE')
    plt.plot(degrees, mse_val,   's-', label='Validation MSE')
    plt.plot(degrees, mse_test,  'd-', label='Test MSE')
    plt.xlabel('Polynomial degree p')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs. Polynomial degree')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
