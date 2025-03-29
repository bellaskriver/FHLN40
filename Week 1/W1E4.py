import numpy as np # imports the library numpy and changes the name to np

def example_func(x): # creates a function
    y = np.zeros(len(x)) # creates a vector y of zeros with the same length as x
    for i in range(len(x)): # for-loop iterates over the length of x, which means the number of elements in x
                if x[i] < 0.0: # if statement checks if the value of the element in x is less than 0
                    y[i] = np.sin(x[i]) # if the value is less than 0, y is equalt to the sine of the value in x
                else:
                    y[i] = np.cos(x[i]) # if the value is greater than or equal to 0, y is equal to the cosine of the value in x
    y_sum_abs = 0.0 # creates a variable y_sum_abs and sets it to 0
    while y_sum_abs < 2.0: # while-loop checks if the sum of the absolute values of y is less than 2
        y_sum_abs += np.abs(y[i]) # adds the absolute value of y to y_sum_abs
    return y, y_sum_abs # returns y and y_sum_abs

# main program
a = np.linspace(-2.0, 2.0, 4) # creates a vector a with 4 elements, linearly spaced between -2 and 2
b, b_sum_abs = example_func(a) # calls the function example_func with a as input and assigns the output to b and b_sum_abs
print("b_sum_abs:", b_sum_abs) # prints the value of b_sum_abs
print("a:", a) # prints the value of a