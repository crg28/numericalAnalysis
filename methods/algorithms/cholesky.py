import numpy as np

# ---------- Pretty Printers ----------
def fmt(x):
    """Formats complex numbers a + bi, rounding small imaginary parts."""
    if abs(x.imag) < 1e-12:
        return f"{x.real: .6f}"
    if abs(x.real) < 1e-12:
        return f"{x.imag: .6f}i"
    return f"{x.real: .6f}{x.imag:+.6f}i"

def print_matrix(M, name=None):
    if name:
        print(f"\n{name}:")
    for row in M:
        print(" " + "  ".join(fmt(v) for v in row))

# ---------- Factorization ----------
def cholesky_like(A):
    """
    Computes a Cholesky-like LU factorization:
        A = L * U   with L[k,k] = U[k,k] = sqrt(A[k,k])
    Allows complex results when the pivot is negative.
    """
    A = np.array(A, dtype=np.complex128)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.complex128)
    U = np.zeros((n, n), dtype=np.complex128)

    for k in range(n):
        Lkk = np.sqrt(A[k, k])
        L[k, k] = Lkk
        U[k, k] = Lkk

        for j in range(k + 1, n):
            U[k, j] = A[k, j] / Lkk
        for i in range(k + 1, n):
            L[i, k] = A[i, k] / Lkk
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i, j] -= L[i, k] * U[k, j]
    return L, U

# ---------- Solvers ----------
def forward_sub(L, b):
    n = len(b)
    y = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        s = sum(L[i, j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / L[i, i]
    return y

def back_sub(U, y):
    n = len(y)
    x = np.zeros(n, dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        s = sum(U[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i, i]
    return x

# ---------- Full Process ----------
def cholesky_demo(A, b):
    print("Cholesky\n")
    print("Results:\n")

    print("Stage 0\n")
    for row in A:
        print(" " + "  ".join(f"{v: .6f}" for v in row))
    print()

    A = np.array(A, dtype=np.complex128)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.complex128)
    U = np.zeros((n, n), dtype=np.complex128)

    for k in range(n):
        Lkk = np.sqrt(A[k, k])
        L[k, k] = Lkk
        U[k, k] = Lkk

        for j in range(k + 1, n):
            U[k, j] = A[k, j] / Lkk
        for i in range(k + 1, n):
            L[i, k] = A[i, k] / Lkk
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i, j] -= L[i, k] * U[k, j]

        print(f"Stage {k+1}\n")
        print_matrix(L, "L")
        print()
        print_matrix(U, "U")
        print()

    y = forward_sub(L, b)
    x = back_sub(U, y)

    print("\nAfter forward and backward substitution\n")
    print("x:")
    for xi in x:
        print(f"{xi.real:.6f}")
    print("\n___________________________________________________________")

# ---------- Run with the provided data ----------
if __name__ == "__main__":
    A = np.array([
        [4,  -1,   0,  3],
        [1,  15.5, 3,  8],
        [0,  -1.3, -4,  1.1],
        [14,  5,   -2, 30]
    ], dtype=float)
    b = np.array([1, 1, 1, 1], dtype=float)
    cholesky_demo(A, b)
