import numpy as np

# function definition
def example_func(x):
    y = np.zeros(len(x)) # body of the function definition is indented
    for i in range(len(x)):
                if x[i] < 0.0: # body of for-loop is indented
                    y[i] = np.sin(x[i]) # body of if statement is indented
                else:
                    y[i] = np.cos(x[i]) # body of else statement is indented
    y_sum_abs = 0.0
    while y_sum_abs < 2.0:
        y_sum_abs += np.abs(y[i]) # body of while-loop is indented
    return y, y_sum_abs

# main program
a = np.linspace(-2.0, 2.0, 4)
b, b_sum_abs = example_func(a)
print("b_sum_abs:", b_sum_abs)
print("a:", a)