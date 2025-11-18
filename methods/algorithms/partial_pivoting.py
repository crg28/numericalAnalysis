# methods/algorithms/partial_pivoting.py
# === Eliminación Gaussiana con Pivoteo Parcial ===

def exchange_rows(A, r1, r2):
    """Intercambia filas r1 <-> r2 en la matriz aumentada."""
    if r1 != r2:
        A[r1], A[r2] = A[r2], A[r1]


def print_matrix(A, step):
    print(f"\nEtapa {step}:")
    for row in A:
        print("  ".join(f"{val: .6f}" for val in row))


def back_substitution(U, n, tol=1e-14):
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        piv = U[i][i]
        if abs(piv) < tol:
            print("\n❌ ERROR EN SUSTITUCIÓN REGRESIVA\n")
            print("Se encontró pivote cero tras la eliminación.\n")
            return None
        suma = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (U[i][n] - suma) / piv
    return x


def gaussian_partial_pivoting(A, n, tol=1e-14):
    print("\nEliminación Gaussiana con Pivoteo Parcial\n")
    print("Matriz aumentada inicial:")
    print_matrix(A, 0)

    for k in range(n - 1):

        # Buscar mayor |A[i][k]| para i >= k
        pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))

        if abs(A[pivot_row][k]) < tol:
            print("\n❌ MATRIZ SINGULAR\n")
            print(
                f"Error [Etapa {k+1}]: No se encontró pivote válido en la columna {k+1}.\n"
                "El sistema NO tiene solución única.\n"
            )
            return None

        # Intercambiar filas si es necesario
        exchange_rows(A, k, pivot_row)

        pivote = A[k][k]

        # Eliminación
        for i in range(k + 1, n):
            factor = A[i][k] / pivote
            for j in range(k, n + 1):
                A[i][j] -= factor * A[k][j]

        print_matrix(A, k + 1)

    return A


# === Entrada pública usada por Django ===
def gaussian_elimination_partial_pivoting(A, b):
    A = [list(map(float, fila)) for fila in A]
    b = list(map(float, b))
    n = len(A)

    Aug = [A[i] + [b[i]] for i in range(n)]

    U = gaussian_partial_pivoting(Aug, n)
    if U is None:
        return

    x = back_substitution(U, n)
    if x is None:
        return

    print("\n🔹 Solución del sistema:")
    for i, val in enumerate(x, start=1):
        print(f"x{i} = {val:.6f}")
