import numpy as np

# 1. How to create array A of size 15, with all zeros?
A = np.zeros(15)
print(A)
# 2. How to find memory size of array A?
print(A.size * A.itemsize)
# 3. How to create array B with values ranging from 20 to 60?
B = np.arange(20, 60 + 1)
# 4. How to create array C of reversed array of B?
C = B[::-1]
# 5. How to create 4×4 array D with values from 0 to 15 (from top to bottom, left to right)?
D = np.arange(0, 16).reshape(4, 4)
# 6. How to find the dimensions of array E [[3, 4, 5], [6, 7, 8]]?
E = np.array([[3, 4, 5], [6, 7, 8]])
print(E.shape)
# 7. How to find indices for non-zero elements from array F [0, 3, 0, 0, 4, 0]?
F = np.array([0, 3, 0, 0, 4, 0])
print(np.nonzero(F))
# 8. How to create 3×3×3 array G with random values?
G = np.random.random((3, 3, 3))
print(G)
# 9. How to find maximum values in array H [1, 13, 0, 56, 71, 22]?
H = np.array([1, 13, 0, 56, 71, 22])
print(H.max())
# 10. How to find minimum values in array H?
print(H.min())
# 11. How to find mean values of array H?
print(H.mean())
# 12. How to find standard deviation of array H?
print(H.std())
# 13. How to find median in array H?
print(np.median(H))
# 14. How to transpose array D?
print(D.transpose())
# 15. How to append array [4, 5, 6] to array I [1, 2, 3]?
I = np.array([1, 2, 3])
print(np.append(I, [4, 5, 6]))
# 16. How to member-wise add, subtract, multiply and divide two arrays J [1, 2, 3] and K [4, 5, 6]?
J = np.array([1, 2, 3])
K = np.array([4, 5, 6])
print(J + K)
print(J - K)
print(J * K)
print(J / K)
# 17. How to find the total sum of elements of array I?
print(I.sum())
# 18. How to find natural log of array I?
print(np.log(I))
# 19. How to use build an array L with [8, 8, 8, 8, 8] using full/repeat function?
L = np.full(5, 8)
# 20. How to sort array M [2, 5, 7, 3, 6]?
M = np.array([2, 5, 7, 3, 6])
print(np.sort(M))
# 21. How to find the indices of the maximum values in array M?
print(np.argmax(M))
# 22. How to find the indices of the minimum values in array M?
print(np.argmin(M))
# 23. How to find the indices of elements in array M that will be sorted?
print(np.argsort(M))
# 24. How to find the inverse of array N = [[6, 1, 1], [4, -2, 5], [2, 8, 7]] in numpy?
N = np.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
print(np.linalg.inv(N))
# 25. How to find absolute value of array N?
print(np.abs(N))
# 26. How to extract the third column (from all rows) of the array O [[11, 22, 33], [44, 55, 66], [77, 88,
# 99]]?
O = np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
print(O[:, 2])
# 27. How to extract the sub-array consisting of the odd rows and even columns of P [[3, 6, 9, 12], [15,
# 18, 21, 24], [27, 30, 33, 36], [39, 42, 45, 48], [51, 54, 57, 60]] ?
P = np.arange(3, 60 + 1, 3).reshape(5, 4)
print(P[::2, 1::2])
