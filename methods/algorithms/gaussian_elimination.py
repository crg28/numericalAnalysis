# methods/algorithms/gaussian_elimination.py
import numpy as np


def print_matrix(A, step):
    print(f"\nStage {step}:")
    for row in A:
        print("  ".join(f"{val: .6f}" for val in row))


def determinant(M):
    m = len(M)
    if m == 1:
        return M[0][0]
    det = 0.0
    for c in range(m):
        minor = [row[:c] + row[c+1:] for row in M[1:]]
        det += ((-1) ** c) * M[0][c] * determinant(minor)
    return det


def simple_gaussian_elimination(A, n):
    """
    Simple Gaussian Elimination (no pivoting)

    Guarantee:
    - If A[k][k] == 0 → Fatal error: cannot continue.
    """
    # Determinant to verify uniqueness of the solution
    coef = [row[:-1] for row in A]
    detA = determinant(coef)

    if abs(detA) < 1e-12:
        print("\n❌ ERROR: The system does not have a unique solution (det(A) ≈ 0).\n")
        return None

    print("\nSimple Gaussian Elimination\n")
    print("Initial augmented matrix:\n")
    print_matrix(A, 0)

    # Forward elimination
    for k in range(n - 1):
        pivot = A[k][k]

        if abs(pivot) < 1e-14:
            print("\n❌ ERROR IN SIMPLE GAUSSIAN ELIMINATION\n")
            print(
                f"Error [Stage {k+1}]: Zero pivot detected at A[{k+1},{k+1}] = 0.\n"
                "Factorization cannot continue because this method does NOT use pivoting.\n"
                "Partial or total pivoting is recommended for this system.\n"
            )
            return None

        for i in range(k + 1, n):
            m = A[i][k] / pivot
            for j in range(k, n + 1):
                A[i][j] -= m * A[k][j]

        print_matrix(A, k + 1)

    return A


def back_substitution(A, n):
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < 1e-14:
            print("\n❌ ERROR: Division by zero in back substitution.\n")
            return None
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (A[i][n] - s) / A[i][i]
    return x


# ---- Public entry point (Django) ----
def gaussian_elimination(A, b):
    A = [list(map(float, row)) for row in A]
    b = list(map(float, b))
    n = len(A)

    Aug = [A[i] + [b[i]] for i in range(n)]

    A_tri = simple_gaussian_elimination(Aug, n)
    if A_tri is None:
        return

    x = back_substitution(A_tri, n)
    if x is None:
        return

    print("\n🔹 System solution:")
    for i, xi in enumerate(x, start=1):
        print(f"x{i} = {xi:.6f}")
