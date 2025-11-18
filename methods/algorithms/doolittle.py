# doolittle.py
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

# ---------- Doolittle (diag(L) = 1), with pivot verification U[k,k] ----------
def doolittle_lu(A: np.ndarray):
    """
    Doolittle LU: A = L * U with diag(L) = 1.

    Guarantees:
    - At each stage k:
        1. U[k,k] is computed.
        2. U[k,k] is checked to ensure it is not zero.
        3. If U[k,k] ≈ 0, factorization is aborted.
    """
    A = np.array(A, dtype=np.complex128)
    n = A.shape[0]
    L = np.eye(n, dtype=np.complex128)
    U = np.eye(n, dtype=np.complex128)

    tol_piv = 1e-14

    for k in range(n):
        # 1️⃣ Row k of U
        for j in range(k, n):
            U[k, j] = A[k, j] - sum(L[k, p] * U[p, j] for p in range(k))

        # 🚨 Pivot U[k,k] check
        if abs(U[k, k]) < tol_piv:
            stage = k + 1
            raise ValueError(
                f"Error [Stage {stage}]: Division by zero. "
                "The computed diagonal element u_kk is zero. "
                "Doolittle factorization cannot be completed."
            )

        # 2️⃣ Column k of L
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - sum(L[i, p] * U[p, k] for p in range(k))) / U[k, k]

    return L, U


def doolittle_demo(A: np.ndarray, b: np.ndarray) -> None:
    print("Doolittle\n")
    print("Results:\n")

    print("Stage 0\n")
    for row in A:
        print(" " + "  ".join(f"{v: .6f}" for v in row))
    print()

    Awork = np.array(A, dtype=np.complex128)
    n = Awork.shape[0]
    L = np.eye(n, dtype=np.complex128)
    U = np.eye(n, dtype=np.complex128)

    tol_piv = 1e-14

    for k in range(n):

        # 1️⃣ Row k of U
        for j in range(k, n):
            U[k, j] = Awork[k, j] - sum(L[k, p] * U[p, j] for p in range(k))

        # 🚨 Pivot check
        if abs(U[k, k]) < tol_piv:
            stage = k + 1
            print("\nNUMERICAL ERROR IN DOOLITTLE FACTORIZATION\n")
            print(
                f"Error [Stage {stage}]: Division by zero. "
                "The computed diagonal element u_kk is zero. "
                "Doolittle factorization cannot be completed."
            )
            print("\n____________________________________________________________________________")
            return

        # 2️⃣ Column k of L
        for i in range(k + 1, n):
            L[i, k] = (Awork[i, k] - sum(L[i, p] * U[p, k] for p in range(k))) / U[k, k]

        print(f"Stage {k+1}\n")
        print_matrix(L, "L")
        print()
        print_matrix(U, "U")
        print()

    # Solution
    y = forward_sub(L, b.astype(np.complex128))
    x = back_sub(U, y)

    print("\nAfter forward and backward substitution\n")
    print("x:")
    for xi in x:
        print(f"{xi.real:.6f}")
    print("\n____________________________________________________________________________")
