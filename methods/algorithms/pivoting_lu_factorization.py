import numpy as np

def print_matrix(M):
    for row in M:
        print(" " + "  ".join(f"{v: .6f}" for v in row))
    print()

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]
    return y

def back_substitution(U, y):
    n = len(y)
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (y[i] - s) / U[i, i]
    return x

def lu_parcial_demo(A, b):
    """
    LU con pivoteo parcial (PA = LU), mostrando:
      - Etapas (A transformándose por eliminación)
      - L (diagonal unitaria)
      - U (rellenada por filas al final de cada etapa para coincidir con la tabla esperada)
      - P (matriz de permutación acumulada)

    También resuelve Ax=b vía PA=LU.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    Awork = A.copy()
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    P = np.eye(n, dtype=float)

    print("LU con pivoteo parcial\n")
    print("Resultados:\n")

    # Etapa 0: matriz original
    print("Etapa 0\n")
    print_matrix(Awork)

    for k in range(n - 1):
        # ---------- Pivoteo parcial: elegir fila con |A[i,k]| máxima ----------
        pivot_row = k + np.argmax(np.abs(Awork[k:, k]))
        if abs(Awork[pivot_row, k]) < 1e-12:
            raise ValueError(f"Pivote ~ 0 en columna {k}")

        if pivot_row != k:
            # Permutar filas en A
            Awork[[k, pivot_row], :] = Awork[[pivot_row, k], :]
            # Permutar filas correspondientes en P
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            # MUY IMPORTANTE: permutar en L las columnas < k (las ya fijadas)
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # ---------- Eliminación por debajo del pivote ----------
        piv = Awork[k, k]
        for i in range(k + 1, n):
            m = Awork[i, k] / piv
            L[i, k] = m
            Awork[i, k:] -= m * Awork[k, k:]

        # ---------- Relleno de U para coincidir con la tabla esperada ----------
        # Fila k de U queda definitiva
        U[k, k:] = Awork[k, k:]
        # Exponer la fila k+1 que acaba de formarse (como en las salidas esperadas)
        if k + 1 < n - 1:
            U[k + 1, k + 1:] = Awork[k + 1, k + 1:]
        elif k + 1 == n - 1:
            U[n - 1, n - 1] = Awork[n - 1, n - 1]

        # ---------- Impresión de la etapa ----------
        print(f"Etapa {k + 1}\n")
        print_matrix(Awork)
        print("L:")
        print_matrix(L)
        print("U:")
        print_matrix(U)
        print("P:")
        print_matrix(P)

    # Resolver PA = LU -> Ax = b
    # Paso 1: Pb
    bp = P @ b
    # Paso 2: L y = Pb
    y = forward_substitution(L, bp)
    # Paso 3: U x = y
    x = back_substitution(U, y)

    print("\n\nDespués de aplicar sustitución progresiva y regresiva\n")
    print("x:")
    for xi in x:
        print(f"{xi:.6f}")

    return Awork, L, U, P, x


# ---- Datos de prueba (los mismos que en tus tablas) ----
if __name__ == "__main__":
    A = [
        [4,  -1,   0,  3],
        [1,  15.5, 3,  8],
        [0,  -1.3, -4,  1.1],
        [14,  5,   -2, 30],
    ]
    b = [1, 1, 1, 1]

    lu_parcial_demo(A, b)
