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
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        piv = A[i][i]
        if abs(piv) < tol:
            raise ZeroDivisionError(f"Pivot ~0 at position ({i},{i})")
        s = 0.0
        for j in range(i+1, n):
            s += A[i][j]*x[j]
        x[i] = (A[i][n] - s)/piv
    return x

def partial_pivoting(A, n, tol=1e-12):
    """Gaussian elimination with partial pivoting."""
    print("Gaussian elimination with partial pivoting\n\nResults:")
    print_matrix(A, step=0)

    for k in range(n-1):
        # Find pivot row with max |A[i,k]| (for i ≥ k)
        pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[pivot_row][k]) < tol:
            raise ValueError(f"Pivot ~0 in column {k}")
        if pivot_row != k:
            exchange_rows(A, k, pivot_row)

        # Elimination
        piv = A[k][k]
        for i in range(k+1, n):
            factor = A[i][k]/piv
            for j in range(k, n+1):
                A[i][j] -= factor*A[k][j]

        print_matrix(A, step=k+1)
    return A

def read_system():
    """Read n, matrix A and vector b, build augmented matrix."""
    n = int(input("Enter n (system size): ").strip())
    print(f"Enter matrix A ({n}x{n}), one row per line:")
    Acoef = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        if len(row) != n:
            raise ValueError(f"Expected {n} values in row {i+1}.")
        Acoef.append(row)
    b = list(map(float, input("Enter vector b (n values): ").split()))
    if len(b) != n:
        raise ValueError(f"Expected {n} values in b.")
    Aaug = [Acoef[i] + [b[i]] for i in range(n)]
    return Aaug, n

def main():
    try:
        Aaug, n = read_system()
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
