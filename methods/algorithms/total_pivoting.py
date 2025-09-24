from sympy import symbols, sympify, lambdify
import numpy as np

def max_in_column(A, col, one_based=False):
    A = np.asarray(A, dtype=float)
    j = col-1 if one_based else col
    if j < 0 or j >= A.shape[1]:
        raise IndexError("Columnn is not in the range.")
    return float(A[:, j].max()), (col if one_based else j)

def exchange_lines(A, l1, l2):
    A[l1], A[l2] = A[l2], A[l1]

def read_augmented_matrix():
    n = int(input("Enter the number of unknowns: "))
    print(f"Enter the augmented matrix row by row (each row with {n+1} values separated by spaces):")
    A = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        if len(row) != n+1:
            print(f"Error: expected {n+1} values, but got {len(row)}.")
            return None, None
        A.append(row)
    return A, n

def print_matrix(A, step):
    print(f"\nStep {step}:")
    for row in A:
        print("  ".join(f"{val:10.4f}" for val in row))

def determinant(M):
    # Recursive calculation using cofactors
    m = len(M)
    if m == 1:
        return M[0][0]
    det = 0.0
    for c in range(m):
        minor = [row[:c] + row[c+1:] for row in M[1:]]
        det += ((-1)**c) * M[0][c] * determinant(minor)
    return det

def back_substitution(A, n):
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        if abs(A[i][i]) < 1e-12:
            print(f"Division by zero in back substitution at position ({i},{i}).")
            return None
        sum_ax = sum(A[i][j]*x[j] for j in range(i+1, n))
        x[i] = (A[i][n] - sum_ax) / A[i][i]
    return x


def total_pivoting(A, n):
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