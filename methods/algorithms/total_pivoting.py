def exchange_lines(A, r1, r2):
    """Swap rows r1 <-> r2 (in place)."""
    if r1 != r2:
        A[r1], A[r2] = A[r2], A[r1]

def exchange_columns(A, c1, c2):
    """Swap columns c1 <-> c2 (in place) in an augmented matrix."""
    if c1 != c2:
        n = len(A)
        for i in range(n):
            A[i][c1], A[i][c2] = A[i][c2], A[i][c1]

def read_augmented_matrix():
    n = int(input("Enter the number of unknowns: "))
    print(f"Enter the augmented matrix row by row (each row with {n+1} values separated by spaces):")
    A = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        if len(row) != n + 1:
            print(f"Error: expected {n+1} values, but got {len(row)}.")
            return None, None
        A.append(row)
    return A, n

def print_matrix(A, step):
    print(f"\nStep {step}:")
    for row in A:
        for number in row:
            print(number, " ", end = ' ')
        print("")

def back_substitution(U_aug, n, tol=1e-12):
    """
    U_aug is upper-triangular augmented matrix (n x (n+1)).
    Returns x aligned to the CURRENT column order (after any column swaps).
    """
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        piv = U_aug[i][i]
        if abs(piv) < tol:
            print(f"Division by zero in back substitution at position ({i},{i}).")
            return None
        s = sum(U_aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (U_aug[i][n] - s) / piv
    return x

def total_pivoting(A, n, tol=1e-16):
    """
    Gaussian elimination with TOTAL PIVOTING on an augmented matrix A (n x (n+1)).
    Returns (U_aug, perm) where:
      - U_aug is upper-triangular augmented matrix after elimination,
      - perm is the column permutation mapping (size n):
            column j currently corresponds to original variable index perm[j].
    """
    # Permutation vector of columns: initially identity
    perm = list(range(n))

    # Initial state
    print_matrix(A, step=0)

    for k in range(n - 1):
        # 1) Looking for maximum in submatrix A[k:n, k:n]
        max_val = 0.0
        p = k
        q = k
        for i in range(k, n):
            for j in range(k, n):
                val = abs(A[i][j])
                if val > max_val:
                    max_val = val
                    p, q = i, j

        # 2) Checking singularity / pivot ~ 0
        if max_val < tol:
            print(f"All candidates ≈ 0 in submatrix [{k}:, {k}:]. System may be singular.")
            return None, None

        # 3) Exchange row and columns and update perm
        if p != k:
            exchange_lines(A, k, p)
        if q != k:
            exchange_columns(A, k, q)
            # Updating perm: before variable in k, now is in q and viceverse
            perm[k], perm[q] = perm[q], perm[k]

        # 4) Deleting under pivot
        piv = A[k][k]

        for i in range(k + 1, n):
            factor = A[i][k] / piv
            for j in range(k, n + 1): 
                A[i][j] -= factor * A[k][j]

        # Print
        print_matrix(A, step=k + 1)

    return A, perm

def main():
    A, n = read_augmented_matrix()
    if A is None:
        return

    U_aug, perm = total_pivoting(A, n)
    if U_aug is None:
        return
    
    x_perm = back_substitution(U_aug, n)
    if x_perm is None:
        return

    # Using perm to return to original order
    x_final = [0.0] * n
    for j in range(n):
        x_final[perm[j]] = x_perm[j]

    print("\nSolutions of the system (original variable order):")
    for idx, val in enumerate(x_final, start=1):
        print(f"x{idx} = {val:.4f}")

main()