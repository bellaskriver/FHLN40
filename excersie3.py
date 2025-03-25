def calculate_sum():
    print('Sum')
    total_sum = 0
    for i in range(10):
        i=i+1
        total_sum += i**2
    return total_sum


import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.arange(0, 10, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()

# -*- coding: utf-8 -*-

a = [1, 2, 3, 4, 5]

for item in a:
    print(item)


print(calculate_sum())

def fibbonachi(n):
    print('Fibbonachi')
    a = 0
    b = 1
    for i in range(n):
        a, b = b, a + b
    return a

print(fibbonachi(10))
