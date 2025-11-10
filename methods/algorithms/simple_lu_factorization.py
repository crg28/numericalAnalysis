import numpy as numpy

def factorization(A):
    A = np.array(A)
    U = A
    L = np.identity(A.shape[0])

    for k in range(1,n-1):
        #El pivote no puede ser 0
        if U[k,k] < 10^-12:
            print("Pivot = 0")
            break

        for i in range(k+1,n):
            #calculate the multiplier
            multiplier = U[i,k]/U[k,k]

            #store the multiplier in L
            L[i,k] = multiplier

            #elemental operation
            U[i, k:] = U[i, k:] - multiplier * U[k, k:]

    return L, U

def progresive_sustitution(L, b):
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)

    n = b.shape[0]
    


        

