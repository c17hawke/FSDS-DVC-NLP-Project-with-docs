import numpy as np
from scipy.sparse import csr_matrix


A = np.array([
    [1,0,0,0,0,1,0],
    [0,1,0,3,0,0,0],
    [0,0,0,0,1,0,2]
])

print(A)


S = csr_matrix(A)
print(S)
print(type(S))

B = S.todense()
print(B)
