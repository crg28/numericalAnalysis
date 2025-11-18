import numpy as np

# ------------------------------------------------------
# Printing helpers and substitution
# ------------------------------------------------------
def print_matrix(M):
    """
    Print a NumPy matrix with 6 decimal places.
    """
    for row in M:
        print(" " + "  ".join(f"{v: .6f}" for v in row))
    print()


def forward_substitution(L, b):
    """
    Forward substitution: solves L y = b
    where L is lower triangular.
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
    Back substitution: solves U x = y
    where U is upper triangular.
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
# Simple LU factorization (no pivoting)
# ------------------------------------------------------
def lu_factorization(A, b, tol=1e-12):
    """
    Simple LU factorization based on Gaussian elimination WITHOUT pivoting.

    A : coefficient matrix (n x n)
    b : right-hand-side vector (n,)

    Behavior:
      - Builds L (lower triangular, diag = 1) and U (upper triangular)
      - Prints all elimination stages, L and U at each step
      - Checks at each stage k that pivot a_kk != 0 (or not too small).
        If |a_kk| < tol:
            Error [Stage k]: the matrix does NOT admit simple LU;
            the system would require PA = LU.
      - At the end, also checks the last pivot a_nn.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    Awork = A.copy()
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)

    print("Simple LU Factorization (no pivoting)\n")
    print("Results:\n")

    # Stage 0: original matrix
    print("Stage 0\n")
    print_matrix(Awork)

    # Forward elimination
    for k in range(n - 1):
        piv = Awork[k, k]

        # ---------- Pivot zero / too small check ----------
        if abs(piv) < tol:
            stage = k + 1  # 1-based display
            print("❌ ERROR IN SIMPLE LU FACTORIZATION\n")
            print(
                f"Error [Stage {stage}]: Pivot a_kk at position "
                f"({stage},{stage}) is zero (or numerically too close to zero).\n"
                "Gaussian elimination without row swapping cannot continue.\n\n"
                "Conclusion: The matrix does NOT admit simple LU factorization "
                "of the form A = L U.\n"
                "A factorization with row permutation is required: P A = L U."
            )
            return None, None, None, None
        # --------------------------------------------------

        # Elimination below pivot
        for i in range(k + 1, n):
            m = Awork[i, k] / piv
            L[i, k] = m
            Awork[i, k:] -= m * Awork[k, k:]

        # ---- Update U for table-style display ----
        U[k, k:] = Awork[k, k:]
        if k + 1 < n - 1:
            U[k + 1, k + 1:] = Awork[k + 1, k + 1:]
        elif k + 1 == n - 1:
            U[n - 1, n - 1] = Awork[n - 1, n - 1]
        # ------------------------------------------

        # Print stage
        print(f"Stage {k + 1}\n")
        print_matrix(Awork)
        print("L:")
        print_matrix(L)
        print("U:")
        print_matrix(U)

    # ---------- Check last pivot (a_nn) ----------
    last_piv = Awork[n - 1, n - 1]
    if abs(last_piv) < tol:
        stage = n
        print("❌ ERROR IN SIMPLE LU FACTORIZATION\n")
        print(
            f"Error [Stage {stage}]: Last pivot a_nn at position "
            f"({stage},{stage}) is zero (or numerically too close to zero).\n"
            "The matrix does not admit simple LU factorization (A = L U) without "
            "row swapping.\n"
            "A factorization of the form P A = L U is required."
        )
        return None, None, None, None

    U[n - 1, n - 1] = last_piv
    # ---------------------------------------------------

    # Solve A x = b using L and U
    y = forward_substitution(L, b)
    x = back_substitution(U, y)

    print("\nAfter applying forward and backward substitution\n")
    print("x:")
    for xi in x:
        print(f"{xi:.6f}")

    return Awork, L, U, x


# ---- Local test ----
if __name__ == "__main__":
    A = [
        [4,  -1,   0,  3],
        [1,  15.5, 3,  8],
        [0,  -1.3, -4,  1.1],
        [14,  5,   -2, 30],
    ]
    b = [1, 1, 1, 1]

    Awork, L, U, x = lu_factorization(A, b)
