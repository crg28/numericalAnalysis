from sympy import symbols, sympify, lambdify
import numpy as np
from gaussian_elimination import read_augmented_matrix, print_matrix, back_substitution, determinant

def max_in_column(A, col, one_based=False):
    A = np.asarray(A, dtype=float)
    j = col-1 if one_based else col
    if j < 0 or j >= A.shape[1]:
        raise IndexError("Columnn is not in the range.")
    return float(A[:, j].max()), (col if one_based else j)

def exchange_lines(A, l1, l2):
    A[l1], A[l2] = A[l2], A[l1]

def partial_pivoting(A, n):
    # Check determinant of coefficient submatrix
    coef = [row[:-1] for row in A]
    det = determinant(coef)
    if abs(det) < 1e-10:
        print("The system does not have a unique solution (determinant ≈ 0).")
        return None

    print_matrix(A, step=0)

    # Perform n-1 elimination steps
    for k in range(n-1):
        if abs(A[k][k]) < 1e-12:
            print(f"Zero pivot at position ({k},{k}); cannot divide.")
            return None
        for i in range(k+1, n):
            temp_max = max_in_column(A, k)
            if temp_max[1] == k:
                pass
            else:
                exchange_lines(A, temp_max[1], i)
            factor = A[i][k] / A[k][k]
            for j in range(k, n+1):
                A[i][j] -= factor * A[k][j]
                print_matrix(A, step=j)
            

        
        
    return A

def main():
    A, n = read_augmented_matrix()
    if A is None:
        return

    A_tri = partial_pivoting(A, n)
    if A_tri is None:
        return

    solutions = back_substitution(A_tri, n)
    if solutions is None:
        return

    print("\nSolutions of the system:")
    for idx, val in enumerate(solutions, start=1):
        print(f"x{idx} = {val:.4f}")

main()