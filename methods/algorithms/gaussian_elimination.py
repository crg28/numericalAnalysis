# methods/algorithms/gaussian_elimination.py
import numpy as np


def print_matrix(A, step):
    print(f"\nEtapa {step}:")
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
    Eliminación Gaussiana Simple (sin pivoteo)

    Garantía:
    - Si A[k][k] == 0 → Error fatal: no es posible continuar.
    """
    # Determinante para verificar unicidad de la solución
    coef = [row[:-1] for row in A]
    detA = determinant(coef)

    if abs(detA) < 1e-12:
        print("\n❌ ERROR: El sistema no tiene solución única (det(A) ≈ 0).\n")
        return None

    print("\nEliminación Gaussiana Simple\n")
    print("Matriz aumentada inicial:\n")
    print_matrix(A, 0)

    # Eliminación hacia adelante
    for k in range(n - 1):
        pivote = A[k][k]

        if abs(pivote) < 1e-14:
            print("\n❌ ERROR EN ELIMINACIÓN GAUSSIANA SIMPLE\n")
            print(
                f"Error [Etapa {k+1}]: Pivote nulo detectado en A[{k+1},{k+1}] = 0.\n"
                "La factorización no puede continuar porque este método NO usa pivoteo.\n"
                "Se recomienda usar pivoteo parcial o total para este sistema.\n"
            )
            return None

        for i in range(k + 1, n):
            m = A[i][k] / pivote
            for j in range(k, n + 1):
                A[i][j] -= m * A[k][j]

        print_matrix(A, k + 1)

    return A


def back_substitution(A, n):
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < 1e-14:
            print("\n❌ ERROR: División por cero en sustitución regresiva.\n")
            return None
        suma = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (A[i][n] - suma) / A[i][i]
    return x


# ---- Entrada pública (Django) ----
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

    print("\n🔹 Solución del sistema:")
    for i, xi in enumerate(x, start=1):
        print(f"x{i} = {xi:.6f}")
