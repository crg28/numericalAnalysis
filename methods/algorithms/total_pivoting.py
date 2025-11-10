# gaussian_elimination_total_pivoting.py

# === Gaussian Elimination with Total Pivoting ===
# Output format follows your required example.

def exchange_rows(A, r1, r2):
    """Swap rows r1 <-> r2 in the augmented matrix."""
    if r1 != r2:
        A[r1], A[r2] = A[r2], A[r1]

def exchange_columns(A, c1, c2):
    """Swap columns c1 <-> c2 (in the coefficient part, not the last column b)."""
    if c1 != c2:
        n = len(A)
        for i in range(n):
            A[i][c1], A[i][c2] = A[i][c2], A[i][c1]

def print_matrix(A, step):
    print(f"\nStep {step}\n")
    for row in A:
        print(" ".join(f"{v: .6f}" for v in row))

def back_substitution(U_aug, n, tol=1e-12):
    """Backward substitution on upper triangular augmented matrix."""
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        piv = U_aug[i][i]
        if abs(piv) < tol:
            raise ZeroDivisionError(f"Pivot ~ 0 at position ({i},{i})")
        s = 0.0
        for j in range(i + 1, n):
            s += U_aug[i][j] * x[j]
        x[i] = (U_aug[i][n] - s) / piv
    return x

def total_pivoting(A, n, tol=1e-16):
    """
    Gaussian elimination with TOTAL PIVOTING on augmented matrix A (n x (n+1)).
    Returns (U_aug, perm) where perm maps current column -> original variable index.
    """
    perm = list(range(n))  # identity permutation

    print("Gaussian elimination with total pivoting\n\nResults:")
    print_matrix(A, step=0)

    for k in range(n - 1):
        # Find maximum in submatrix A[k:n, k:n]
        max_val, p, q = 0.0, k, k
        for i in range(k, n):
            for j in range(k, n):
                val = abs(A[i][j])
                if val > max_val:
                    max_val, p, q = val, i, j

        if max_val < tol:
            raise ValueError(f"Submatrix {k} is singular or nearly singular.")

        # Swap rows and columns
        if p != k:
            exchange_rows(A, k, p)
        if q != k:
            exchange_columns(A, k, q)
            perm[k], perm[q] = perm[q], perm[k]

        # Elimination
        piv = A[k][k]
        for i in range(k + 1, n):
            factor = A[i][k] / piv
            for j in range(k, n + 1):
                A[i][j] -= factor * A[k][j]

        print_matrix(A, step=k + 1)

    return A, perm

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
        U_aug, perm = total_pivoting(Aaug, n)
        x_perm = back_substitution(U_aug, n)

        # Map solution back to original variable order
        x_final = [0.0] * n
        for j in range(n):
            x_final[perm[j]] = x_perm[j]

        print("\n\nAfter applying back substitution\n")
        print("x:")
        for xi in x_final:
            print(f"{xi:.6f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
