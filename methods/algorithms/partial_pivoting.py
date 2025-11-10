# gaussian_elimination_partial_pivoting.py

# === Gaussian Elimination with PARTIAL Pivoting ===

def exchange_rows(A, r1, r2):
    """Swap rows r1 <-> r2 in the augmented matrix."""
    if r1 != r2:
        A[r1], A[r2] = A[r2], A[r1]

def print_matrix(A, step):
    print(f"\nStep {step}\n")
    for row in A:
        print(" ".join(f"{v: .6f}" for v in row))

def back_substitution(A, n, tol=1e-12):
    """Perform back substitution on an upper-triangular augmented matrix."""
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        piv = A[i][i]
        if abs(piv) < tol:
            raise ZeroDivisionError(f"Pivot ~0 at position ({i},{i})")
        s = 0.0
        for j in range(i + 1, n):
            s += A[i][j] * x[j]
        x[i] = (A[i][n] - s) / piv
    return x

def partial_pivoting(A, n, tol=1e-12):
    """Gaussian elimination with partial pivoting."""
    print("Gaussian elimination with partial pivoting\n\nResults:")
    print_matrix(A, step=0)

    for k in range(n - 1):
        # Find pivot row with max |A[i,k]| (for i ≥ k)
        pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[pivot_row][k]) < tol:
            raise ValueError(f"Pivot ~0 in column {k}")
        if pivot_row != k:
            exchange_rows(A, k, pivot_row)

        # Elimination
        piv = A[k][k]
        for i in range(k + 1, n):
            factor = A[i][k] / piv
            for j in range(k, n + 1):
                A[i][j] -= factor * A[k][j]

        print_matrix(A, step=k + 1)
    return A

def main():
    # ---------- Fixed test system (no input) ----------
    Acoef = [
        [2.0,  -1.0,  0.0,  3.0],
        [1.0,   0.5,  3.0,  8.0],
        [0.0,  13.0, -2.0, 11.0],
        [14.0,  5.0, -2.0,  3.0],
    ]
    b = [1.0, 1.0, 1.0, 1.0]
    n = len(Acoef)

    # Build augmented matrix [A | b]
    Aaug = [row[:] + [bi] for row, bi in zip(Acoef, b)]

    try:
        U = partial_pivoting(Aaug, n)
        x = back_substitution(U, n)
        print("\n\nAfter applying back substitution\n")
        print("x:")
        for val in x:
            print(f"{val:.6f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
