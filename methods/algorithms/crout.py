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

# ---------- Crout (diag(U) = 1), staged printing ----------
def crout_lu(A: np.ndarray):
    """
    Crout LU factorization: A = L * U with diag(U) = 1.
    We initialize L and U as identities so that untouched diagonal entries
    remain 1.000000 at every stage (matching your reference tables).
    """
    A = np.array(A, dtype=np.complex128)
    n = A.shape[0]
    L = np.eye(n, dtype=np.complex128)  # keep future diagonals as 1 until updated
    U = np.eye(n, dtype=np.complex128)  # diag(U)=1 always

    for k in range(n):
        # Column k of L (including the diagonal)
        for i in range(k, n):
            L[i, k] = A[i, k] - sum(L[i, p] * U[p, k] for p in range(k))

        # Row k of U (to the right of the diagonal); U[k,k] already 1
        for j in range(k + 1, n):
            U[k, j] = (A[k, j] - sum(L[k, p] * U[p, j] for p in range(k))) / L[k, k]

    return L, U

def crout_demo(A: np.ndarray, b: np.ndarray) -> None:
    print("Crout\n")
    print("Results:\n")

    print("Stage 0\n")
    for row in A:
        print(" " + "  ".join(f"{v: .6f}" for v in row))
    print()

    # staged factorization for display
    Awork = np.array(A, dtype=np.complex128)
    n = Awork.shape[0]
    L = np.eye(n, dtype=np.complex128)
    U = np.eye(n, dtype=np.complex128)

    for k in range(n):
        for i in range(k, n):
            L[i, k] = Awork[i, k] - sum(L[i, p] * U[p, k] for p in range(k))
        for j in range(k + 1, n):
            U[k, j] = (Awork[k, j] - sum(L[k, p] * U[p, j] for p in range(k))) / L[k, k]

        print(f"Stage {k+1}\n")
        print_matrix(L, "L")
        print()
        print_matrix(U, "U")
        print()

    # Solve A x = b via LU
    y = forward_sub(L, b.astype(np.complex128))
    x = back_sub(U, y)

    print("\nAfter forward and backward substitution\n")
    print("x:")
    for xi in x:
        print(f"{xi.real:.6f}")
    print("\n____________________________________________________________________________")

# ---------- Run with your provided data ----------
if __name__ == "__main__":
    A = np.array([
        [4,  -1,   0,  3],
        [1,  15.5, 3,  8],
        [0,  -1.3, -4,  1.1],
        [14,  5,   -2, 30]
    ], dtype=float)
    b = np.array([1, 1, 1, 1], dtype=float)
    crout_demo(A, b)
