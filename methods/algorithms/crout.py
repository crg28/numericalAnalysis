# crout.py
import numpy as np

# ---------- Pretty Printers ----------
def fmt(x: complex) -> str:
    if abs(x.imag) < 1e-12:
        return f"{x.real: .6f}"
    if abs(x.real) < 1e-12:
        return f"{x.imag: .6f}i"
    return f"{x.real: .6f}{x.imag:+.6f}i"

def print_matrix(M: np.ndarray, name: str | None = None) -> None:
    if name:
        print(f"\n{name}:")
    for row in M:
        print(" " + "  ".join(fmt(v) for v in row))

# ---------- Solvers ----------
def forward_sub(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    y = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        s = sum(L[i, j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / L[i, i]
    return y

def back_sub(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    x = np.zeros(n, dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        s = sum(U[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i, i]
    return x

# ---------- Crout (diag(U) = 1), con comprobación de l_kk ----------
def crout_lu(A: np.ndarray):
    """
    Factorización Crout: A = L * U con diag(U) = 1.

    Garantías:
    - Calcula columnas de L y filas de U secuencialmente por etapa k.
    - En cada etapa k:
        1. Calcula l_kk.
        2. Verifica que l_kk != 0 (dentro de una tolerancia).
        3. Si l_kk es ~0, lanza ValueError con mensaje descriptivo.
    """
    A = np.array(A, dtype=np.complex128)
    n = A.shape[0]
    L = np.eye(n, dtype=np.complex128)  # L se irá llenando; diag(L) se actualiza
    U = np.eye(n, dtype=np.complex128)  # diag(U) = 1

    tol_piv = 1e-14  # tolerancia para considerar l_kk como cero

    for k in range(n):
        # ---------- 1. Columna k de L ----------
        # Primero el elemento diagonal l_kk
        L[k, k] = A[k, k] - sum(L[k, p] * U[p, k] for p in range(k))

        # Comprobación de pivote cero en Crout
        if abs(L[k, k]) < tol_piv:
            etapa = k + 1  # etapas en 1..n
            msg = (
                f"Error [Stage {etapa}]: Division by zero. "
                "The computed diagonal element l_kk is zero. "
                "The Crout factorization cannot be completed."

            )
            raise ValueError(msg)

        # Luego el resto de la columna k (i > k)
        for i in range(k + 1, n):
            L[i, k] = A[i, k] - sum(L[i, p] * U[p, k] for p in range(k))

        # ---------- 2. Fila k de U (a la derecha de la diagonal) ----------
        for j in range(k + 1, n):
            U[k, j] = (A[k, j] - sum(L[k, p] * U[p, j] for p in range(k))) / L[k, k]

    return L, U

def crout_demo(A: np.ndarray, b: np.ndarray) -> None:
    """
    Demo version of the Crout method, which:
    - Prints each stage with L and U.
    - Performs forward and backward substitution.
    - Detects when l_kk = 0 at any stage and shows an error message
    instead of continuing with the factorization.
    """

    print("Crout\n")
    print("Results:\n")

    print("Stage 0\n")
    for row in A:
        print(" " + "  ".join(f"{v: .6f}" for v in row))
    print()

    Awork = np.array(A, dtype=np.complex128)
    n = Awork.shape[0]
    L = np.eye(n, dtype=np.complex128)
    U = np.eye(n, dtype=np.complex128)

    tol_piv = 1e-14  # tolerancia para considerar l_kk como cero

    # ---------- Factorización por etapas ----------
    for k in range(n):
        # 1. Columna k de L
        #    Primero el diagonal l_kk
        L[k, k] = Awork[k, k] - sum(L[k, p] * U[p, k] for p in range(k))

        #    Comprobación de pivote cero
        if abs(L[k, k]) < tol_piv:
            etapa = k + 1
            print("\nERROR NUMÉRICO EN FACTORIZACIÓN CROUT\n")
            print(
                f"Error [Stage {etapa}]: Division by zero. "
                "The computed diagonal element l_kk is zero. "
                "The Crout factorization cannot be completed."

            )
            # No seguimos calculando ni resolviendo
            print("\n____________________________________________________________________________")
            return

        #    Luego el resto de la columna (i > k)
        for i in range(k + 1, n):
            L[i, k] = Awork[i, k] - sum(L[i, p] * U[p, k] for p in range(k))

        # 2. Fila k de U (a la derecha del diagonal)
        for j in range(k + 1, n):
            U[k, j] = (Awork[k, j] - sum(L[k, p] * U[p, j] for p in range(k))) / L[k, k]

        # 3. Imprimir etapa
        print(f"Stage {k+1}\n")
        print_matrix(L, "L")
        print()
        print_matrix(U, "U")
        print()

    # ---------- Resolver A x = b via LU ----------
    y = forward_sub(L, b.astype(np.complex128))
    x = back_sub(U, y)

    print("\nAfter forward and backward substitution\n")
    print("x:")
    for xi in x:
        # Para mostrar solo parte real si es prácticamente real
        if abs(xi.imag) < 1e-12:
            print(f"{xi.real:.6f}")
        else:
            print(f"{xi.real:.6f} + {xi.imag:.6f}i")
    print("\n____________________________________________________________________________")

