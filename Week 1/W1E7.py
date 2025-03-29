import numpy as np

# array creation
a = np.zeros((5,5)) 		  # array filled with zeros (5, 5)
b = np.ones(5)	 		      # array filled with ones (5)
c = np.array([1,2,3,4,5])	# array created from a list (5)
d = np.linspace(0,1,10)   # linearly spaced array in range [0,1](10)

# array transformations
bc = np.concatenate((b,c), axis=0)# concatenation of arrays b and c
a_reshape = np.reshape(a, 25)	# reshape into an array of dimension (25)
a_flat = a.flatten()		# flattens an array the dimension (25)

# indexing in Python the index starts at 0 and thus ends at n-1
c = np.array([1,2,3,4,5])
c[-1]		# last element
c[:2]		# the first two elements (excluding i=2)
c[2:]		# elements from the third elements onward (including i=2)
c[1:3]	# second element until third element (i=1,2)
c[::2]	# every second element starting from i=0
c[::-1]	# flips the order of the elements

# linear algebra
np.dot(b,c)		  # scalar product of vector b and c
np.matmul(a,b)	# matrix multiplication, also a @ b possible

# Elementwise math operations (most math expressions from the math library are available)
np.sin(d*np.pi)	# elementwise application of the sine
b*c 			      # elementwise multiplication of vectors b and c

# matrix operations
A = np.array([[1, 2, 3], [4, 5, 6]])
print("A.shape:", A.shape)
B = np.array([[1, 2], [3, 4], [5, 6]])
print("B.shape:", B.shape)
C = A @ B
print(C)
print("C.shape:", C.shape)
