def calculate_sum():
    print('Sum')
    total_sum = 0
    for i in range(10):
        i=i+1
        total_sum += i**2
    return total_sum

print(calculate_sum())
