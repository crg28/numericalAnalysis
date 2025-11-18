import numpy as np

def print_matrix(M):
    """
    Imprime una matriz con formato bonito.
    """
    for row in M:
        print(" " + "  ".join(f"{v: .6f}" for v in row))
    print()


def forward_substitution(L, b):
    """
    Sustitución progresiva: resuelve Ly = b
    para L triangular inferior.
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
    Sustitución regresiva: resuelve Ux = y
    para U triangular superior.
    """
    n = len(y)
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (y[i] - s) / U[i, i]
    return x


def pivoting_lu_factorization(A, b):
    """
    LU con pivoteo parcial (PA = LU).

    Garantías (pivoteo + singularidad):

    1. En cada etapa k (k = 0..n-2), se busca el mayor |A[i,k]| con i = k..n-1.
    2. Si ese máximo absoluto es ~0, se imprime un mensaje claro de error:

       ❌ ERROR NUMÉRICO EN FACTORIZACIÓN LU
       La matriz es singular.
       Motivo: No se encontró un pivote no nulo en la columna k+1.
       Etapa donde falló: k+1.

       Y la función retorna:
         (None, None, None, None, None)

    3. Al final de la eliminación, se verifica también el último pivote
       A[n-1,n-1]. Si es ~0, se imprime el mismo tipo de mensaje
       con etapa = n y columna = n, y se retorna todo None.

    Si la matriz NO es singular, imprime las etapas, L, U, P y la solución x.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    Awork = A.copy()
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    P = np.eye(n, dtype=float)

    tol_piv = 1e-14  # tolerancia para considerar pivote ~ 0

    print("LU con pivoteo parcial\n")
    print("Resultados:\n")

    # Etapa 0: matriz original
    print("Etapa 0\n")
    print_matrix(Awork)

    # Etapas de eliminación k = 0..n-2
    for k in range(n - 1):
        # ---------- 1. Pivoteo parcial en la columna k ----------
        # Subcolumna desde la fila k hacia abajo
        col_k = np.abs(Awork[k:, k])
        max_abs = np.max(col_k)

        # Comprobación de singularidad en esta etapa
        if max_abs < tol_piv:
            etapa = k + 1
            col = k + 1
            msg = (
                f"La matriz es singular.\n"
                f"Motivo: No se encontró un pivote no nulo en la columna {col}.\n"
                f"Etapa donde falló: {etapa}."
            )
            print("\n❌ ERROR NUMÉRICO EN FACTORIZACIÓN LU\n")
            print(msg)
            return None, None, None, None, None

        # Fila del pivote (parcial)
        pivot_row = k + np.argmax(col_k)

        # Si la fila de pivote no es la actual, intercambiamos
        if pivot_row != k:
            # Permutar filas en Awork
            Awork[[k, pivot_row], :] = Awork[[pivot_row, k], :]
            # Permutar filas en P
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            # Permutar en L las columnas < k (ya definidas)
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # ---------- 2. Eliminación por debajo del pivote ----------
        piv = Awork[k, k]
        for i in range(k + 1, n):
            m = Awork[i, k] / piv
            L[i, k] = m
            Awork[i, k:] -= m * Awork[k, k:]

        # ---------- 3. Relleno de U para mostrar la etapa ----------
        # La fila k de U queda definitiva
        U[k, k:] = Awork[k, k:]
        # Mostrar algo de la fila k+1, como en tus tablas
        if k + 1 < n - 1:
            U[k + 1, k + 1:] = Awork[k + 1, k + 1:]
        elif k + 1 == n - 1:
            U[n - 1, n - 1] = Awork[n - 1, n - 1]

        # ---------- 4. Impresión de la etapa ----------
        print(f"Etapa {k + 1}\n")
        print_matrix(Awork)
        print("L:")
        print_matrix(L)
        print("U:")
        print_matrix(U)
        print("P:")
        print_matrix(P)

    # ---------- Chequeo final del último pivote ----------
    last_piv = Awork[n - 1, n - 1]
    if abs(last_piv) < tol_piv:
        etapa = n
        col = n
        msg = (
            f"La matriz es singular.\n"
            f"Motivo: No se encontró un pivote no nulo en la columna {col}.\n"
            f"Etapa donde falló: {etapa}."
        )
        print("\nERROR NUMÉRICO EN FACTORIZACIÓN LU\n")
        print(msg)
        return None, None, None, None, None

    # ---------- Resolución del sistema PA = LU ----------
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

