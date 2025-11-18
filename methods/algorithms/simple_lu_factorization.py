import numpy as np

# ------------------------------------------------------
# Helpers de impresión y sustitución
# ------------------------------------------------------
def print_matrix(M):
    """
    Imprime una matriz NumPy con 6 decimales.
    """
    for row in M:
        print(" " + "  ".join(f"{v: .6f}" for v in row))
    print()


def forward_substitution(L, b):
    """
    Sustitución progresiva: resuelve L y = b
    donde L es triangular inferior.
    """
    n = len(b)
    y = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]
    return y


def back_substitution(U, y):
    """
    Sustitución regresiva: resuelve U x = y
    donde U es triangular superior.
    """
    n = len(y)
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (y[i] - s) / U[i, i]
    return x


# ------------------------------------------------------
# Factorización LU simple (sin pivoteo)
# ------------------------------------------------------
def lu_factorization(A, b, tol=1e-12):
    """
    Factorización LU simple basada en eliminación gaussiana SIN pivoteo.

    A: matriz de coeficientes (n x n)
    b: vector independiente (n,)

    Comportamiento:
      - Construye L (triangular inferior, diag = 1) y U (triangular superior)
      - Imprime las etapas de la eliminación, L y U en cada una
      - Verifica en cada etapa k que el pivote a_kk != 0 (ni muy pequeño).
        Si |a_kk| < tol:
          Error [Etapa k]: la matriz no admite LU simple; se requeriría PA = LU.
      - Al final, verifica también el último pivote a_nn.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    Awork = A.copy()
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)

    print("Factorización LU simple (sin pivoteo)\n")
    print("Resultados:\n")

    # Etapa 0: matriz original
    print("Etapa 0\n")
    print_matrix(Awork)

    # Eliminación hacia adelante
    for k in range(n - 1):
        piv = Awork[k, k]

        # ---------- Comprobación de pivote cero / muy pequeño ----------
        if abs(piv) < tol:
            etapa = k + 1  # para mostrar 1-based
            print("❌ ERROR EN FACTORIZACIÓN LU SIMPLE\n")
            print(
                f"Error [Etapa {etapa}]: El pivote a_kk en la posición "
                f"({etapa},{etapa}) es cero (o numéricamente muy cercano a cero).\n"
                "La eliminación gaussiana sin intercambio de renglones no puede "
                "continuar.\n\n"
                "Conclusión: La matriz NO admite factorización LU simple de la "
                "forma A = L U.\n"
                "Se requeriría una factorización con permutación de filas: P A = L U."
            )
            return None, None, None, None
        # ----------------------------------------------------------------

        # Eliminación por debajo del pivote
        for i in range(k + 1, n):
            m = Awork[i, k] / piv
            L[i, k] = m
            Awork[i, k:] -= m * Awork[k, k:]

        # ---- Actualización de U para mostrar como en las tablas ----
        # La fila k de U queda definitiva
        U[k, k:] = Awork[k, k:]
        # Exponer parcialmente la siguiente fila (estilo tablas de clase)
        if k + 1 < n - 1:
            U[k + 1, k + 1:] = Awork[k + 1, k + 1:]
        elif k + 1 == n - 1:
            # Última fila: al menos la diagonal
            U[n - 1, n - 1] = Awork[n - 1, n - 1]
        # -------------------------------------------------------------

        # Imprimir etapa
        print(f"Etapa {k + 1}\n")
        print_matrix(Awork)
        print("L:")
        print_matrix(L)
        print("U:")
        print_matrix(U)

    # ---------- Comprobación del último pivote (a_nn) ----------
    last_piv = Awork[n - 1, n - 1]
    if abs(last_piv) < tol:
        etapa = n
        print("❌ ERROR EN FACTORIZACIÓN LU SIMPLE\n")
        print(
            f"Error [Etapa {etapa}]: El último pivote a_nn en la posición "
            f"({etapa},{etapa}) es cero (o numéricamente muy cercano a cero).\n"
            "La matriz no admite factorización LU simple (A = L U) sin "
            "intercambio de filas.\n"
            "Se requeriría una factorización del tipo P A = L U."
        )
        return None, None, None, None
    # Aseguramos que U[n-1, n-1] sea consistente
    U[n - 1, n - 1] = last_piv
    # ---------------------------------------------------

    # Resolver A x = b usando L y U
    y = forward_substitution(L, b)
    x = back_substitution(U, y)

    print("\nDespués de aplicar sustitución progresiva y regresiva\n")
    print("x:")
    for xi in x:
        print(f"{xi:.6f}")

    return Awork, L, U, x


# ---- Prueba local ----
if __name__ == "__main__":
    A = [
        [4,  -1,   0,  3],
        [1,  15.5, 3,  8],
        [0,  -1.3, -4,  1.1],
        [14,  5,   -2, 30],
    ]
    b = [1, 1, 1, 1]

    Awork, L, U, x = lu_factorization(A, b)
