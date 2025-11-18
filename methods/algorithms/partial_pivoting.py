# methods/algorithms/partial_pivoting.py
# === Gaussian Elimination with Partial Pivoting ===

def exchange_rows(A, r1, r2):
    """Swap rows r1 <-> r2 in the augmented matrix."""
    if r1 != r2:
        A[r1], A[r2] = A[r2], A[r1]


def print_matrix(A, step):
    print(f"\nStage {step}:")
    for row in A:
        print("  ".join(f"{val: .6f}" for val in row))


def back_substitution(U, n, tol=1e-14):
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        piv = U[i][i]
        if abs(piv) < tol:
            print("\n❌ ERROR IN BACK SUBSTITUTION\n")
            print("A zero pivot was found after elimination.\n")
            return None
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (U[i][n] - s) / piv
    return x


def gaussian_partial_pivoting(A, n, tol=1e-14):
    print("\nGaussian Elimination with Partial Pivoting\n")
    print("Initial augmented matrix:")
    print_matrix(A, 0)

    for k in range(n - 1):

        # Find largest |A[i][k]| for i >= k
        pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))

        if abs(A[pivot_row][k]) < tol:
            print("\n❌ SINGULAR MATRIX\n")
            print(
                f"Error [Stage {k+1}]: No valid pivot found in column {k+1}.\n"
                "The system does NOT have a unique solution.\n"
            )
            return None

        # Swap rows if needed
        exchange_rows(A, k, pivot_row)

        pivot = A[k][k]

        # Elimination
        for i in range(k + 1, n):
            factor = A[i][k] / pivot
            for j in range(k, n + 1):
                A[i][j] -= factor * A[k][j]

        print_matrix(A, k + 1)

    return A


# === Public entry point used by Django ===
def gaussian_elimination_partial_pivoting(A, b):
    A = [list(map(float, row)) for row in A]
    b = list(map(float, b))
    n = len(A)

    Aug = [A[i] + [b[i]] for i in range(n)]

    U = gaussian_partial_pivoting(Aug, n)
    if U is None:
        return

    x = back_substitution(U, n)
    if x is None:
        return

    print("\n🔹 System solution:")
    for i, val in enumerate(x, start=1):
        print(f"x{i} = {val:.6f}")
