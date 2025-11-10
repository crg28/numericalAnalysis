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

def gaussian_elimination_lu(A, b):
    """
    Simple Gaussian elimination (no pivoting) that constructs L and U step-by-step
    and prints stages exactly like the expected output.

    Key detail to match the tables:
    - After stage k, we set U[k, k:] from the current Awork row k,
      AND also U[k+1, k+1:] from the just-formed row k+1 (when it exists).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    Awork = A.copy()
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)

    print("LU with Simple Gaussian Elimination:\n")
    print("Results:\n")

    # Stage 0: print the initial A
    print("Stage 0\n")
    print_matrix(Awork)

    for k in range(n - 1):
        piv = Awork[k, k]
        if abs(piv) < 1e-12:
            raise ValueError(f"Zero pivot at row {k}; pivoting would be required.")

        # Eliminate below the pivot
        for i in range(k + 1, n):
            m = Awork[i, k] / piv
            L[i, k] = m
            Awork[i, k:] -= m * Awork[k, k:]

        # ---- U UPDATE TO MATCH EXPECTED TABLES ----
        # Row k of U is now final:
        U[k, k:] = Awork[k, k:]
        # Also expose the just-formed next row in U at the same stage:
        if k + 1 < n - 1:
            U[k + 1, k + 1:] = Awork[k + 1, k + 1:]
        elif k + 1 == n - 1:
            # Last row: only the diagonal entry is revealed at this stage.
            U[n - 1, n - 1] = Awork[n - 1, n - 1]
        # -------------------------------------------

        # Print stage block
        print(f"Stage {k + 1}\n")
        print_matrix(Awork)
        print("L:")
        print_matrix(L)
        print("U:")
        print_matrix(U)

    # Solve with forward & backward substitution
    y = forward_substitution(L, b)
    x = back_substitution(U, y)

    print("\nAfter applying forward and backward substitution\n")
    print("x:")
    for xi in x:
        print(f"{xi:.6f}")

    return Awork, L, U, x


# ---- Test Data ----
if __name__ == "__main__":
    A = [
        [4,  -1,   0,  3],
        [1,  15.5, 3,  8],
        [0,  -1.3, -4,  1.1],
        [14,  5,   -2, 30],
    ]
    b = [1, 1, 1, 1]

    gaussian_elimination_lu(A, b)
