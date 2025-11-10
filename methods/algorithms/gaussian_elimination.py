# gaussian_elimination_simple.py

def print_matrix(A, step):
    print(f"\nStep {step}:")
    for row in A:
        print("  ".join(f"{val:10.4f}" for val in row))

def determinant(M):
    # Recursive cofactor expansion (OK for small n)
    m = len(M)
    if m == 1:
        return M[0][0]
    det = 0.0
    for c in range(m):
        minor = [row[:c] + row[c+1:] for row in M[1:]]
        det += ((-1) ** c) * M[0][c] * determinant(minor)
    return det

def simple_gaussian_elimination(A, n):
    # Check determinant of the coefficient submatrix
    coef = [row[:-1] for row in A]
    det = determinant(coef)
    if abs(det) < 1e-10:
        print("The system does not have a unique solution (determinant ≈ 0).")
        return None

    print_matrix(A, step=0)

    # Forward elimination (no pivoting)
    for k in range(n - 1):
        if abs(A[k][k]) < 1e-12:
            print(f"Zero pivot at position ({k},{k}); cannot divide.")
            return None

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n + 1):
                A[i][j] -= factor * A[k][j]

        print_matrix(A, step=k + 1)

    return A

def back_substitution(A, n):
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < 1e-12:
            print(f"Division by zero in back substitution at position ({i},{i}).")
            return None
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (A[i][n] - sum_ax) / A[i][i]
    return x

def main():
    # ---- Fixed test data (no console input) ----
    Acoef = [
        [2.0,  -1.0,  0.0,  3.0],
        [1.0,   0.5,  3.0,  8.0],
        [0.0,  13.0, -2.0, 11.0],
        [14.0,  5.0, -2.0,  3.0],
    ]
    b = [1.0, 1.0, 1.0, 1.0]
    n = len(Acoef)

    # Build augmented matrix [A | b]
    A = [row[:] + [b_i] for row, b_i in zip(Acoef, b)]

    # Eliminate and solve
    A_tri = simple_gaussian_elimination(A, n)
    if A_tri is None:
        return

    solutions = back_substitution(A_tri, n)
    if solutions is None:
        return

    print("\nSolutions of the system:")
    for idx, val in enumerate(solutions, start=1):
        print(f"x{idx} = {val:.6f}")

if __name__ == "__main__":
    main()
