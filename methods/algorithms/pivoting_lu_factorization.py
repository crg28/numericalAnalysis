import numpy as np

def print_matrix(M):
    """
    Print a matrix with clean formatting.
    """
    for row in M:
        print(" " + "  ".join(f"{v: .6f}" for v in row))
    print()


def forward_substitution(L, b):
    """
    Forward substitution: solves Ly = b
    for lower-triangular L.
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
    Back substitution: solves Ux = y
    for upper-triangular U.
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
    LU factorization with partial pivoting (PA = LU).

    Guarantees (pivoting + singularity detection):

    1. At each stage k (k = 0..n-2), the method searches for the largest |A[i,k]|
       with i = k..n-1.

    2. If that maximum absolute value is ~0, an explicit error message is printed:

       ❌ NUMERICAL ERROR IN LU FACTORIZATION
       The matrix is singular.
       Reason: No non-zero pivot was found in column k+1.
       Stage where it failed: k+1.

       And the function returns:
         (None, None, None, None, None)

    3. After elimination, the final pivot A[n-1,n-1] is also checked.
       If it is ~0, the same type of error message is printed
       with stage = n and column = n, and again all None is returned.

    If the matrix is NOT singular, the function prints all stages,
    as well as L, U, P, and the final solution x.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    Awork = A.copy()
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    P = np.eye(n, dtype=float)

    tol_piv = 1e-14  # tolerance to consider a pivot ~ 0

    print("LU with Partial Pivoting\n")
    print("Results:\n")

    # Stage 0: original matrix
    print("Stage 0\n")
    print_matrix(Awork)

    # Elimination stages k = 0..n-2
    for k in range(n - 1):
        # ---------- 1. Partial pivoting in column k ----------
        col_k = np.abs(Awork[k:, k])
        max_abs = np.max(col_k)

        # Singularity check
        if max_abs < tol_piv:
            stage = k + 1
            col = k + 1
            msg = (
                "The matrix is singular.\n"
                f"Reason: No non-zero pivot was found in column {col}.\n"
                f"Stage where it failed: {stage}."
            )
            print("\n❌ NUMERICAL ERROR IN LU FACTORIZATION\n")
            print(msg)
            return None, None, None, None, None

        # Pivot row
        pivot_row = k + np.argmax(col_k)

        # Swap rows if needed
        if pivot_row != k:
            Awork[[k, pivot_row], :] = Awork[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # ---------- 2. Elimination below the pivot ----------
        piv = Awork[k, k]
        for i in range(k + 1, n):
            m = Awork[i, k] / piv
            L[i, k] = m
            Awork[i, k:] -= m * Awork[k, k:]

        # ---------- 3. Fill U for stage display ----------
        U[k, k:] = Awork[k, k:]
        if k + 1 < n - 1:
            U[k + 1, k + 1:] = Awork[k + 1, k + 1:]
        elif k + 1 == n - 1:
            U[n - 1, n - 1] = Awork[n - 1, n - 1]

        # ---------- 4. Print stage ----------
        print(f"Stage {k + 1}\n")
        print_matrix(Awork)
        print("L:")
        print_matrix(L)
        print("U:")
        print_matrix(U)
        print("P:")
        print_matrix(P)

    # ---------- Final pivot check ----------
    last_piv = Awork[n - 1, n - 1]
    if abs(last_piv) < tol_piv:
        stage = n
        col = n
        msg = (
            "The matrix is singular.\n"
            f"Reason: No non-zero pivot was found in column {col}.\n"
            f"Stage where it failed: {stage}."
        )
        print("\n❌ NUMERICAL ERROR IN LU FACTORIZATION\n")
        print(msg)
        return None, None, None, None, None

    # ---------- Solve PA = LU ----------
    bp = P @ b
    y = forward_substitution(L, bp)
    x = back_substitution(U, y)

    print("\n\nAfter applying forward and backward substitution\n")
    print("x:")
    for xi in x:
        print(f"{xi:.6f}")

    return Awork, L, U, P, x
