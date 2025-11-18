import numpy as np

# -------------------------------------------------------------------
# Utilities: spectral radius and diagonal dominance
# -------------------------------------------------------------------
def spectral_radius(T: np.ndarray) -> float:
    """
    Computes the spectral radius of a matrix T (max |eigenvalue|).
    """
    try:
        eigenvalues = np.linalg.eigvals(T)
        return float(np.max(np.abs(eigenvalues)))
    except np.linalg.LinAlgError:
        return np.nan  # In case of numerical error


def is_diagonally_dominant(A: np.ndarray) -> bool:
    """
    Checks row-wise diagonal dominance:
        |a_ii| >= sum_{j != i} |a_ij|  for all rows i.
    (Sufficient condition for convergence of Jacobi / Gauss-Seidel.)
    """
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        sum_off_diag = np.sum(np.abs(A[i, :])) - diag
        if diag < sum_off_diag:
            return False
    return True


# -------------------------------------------------------------------
# Core of the Gauss-Seidel method (matrix form)
# -------------------------------------------------------------------
def gauss_seidel_matricial(A, b, x0, tolerance, niter):
    """
    Matrix implementation of the Gauss-Seidel method:

        A x = b
        A = D - L - U
        (D - L) x^{k+1} = U x^{k} + b
        x^{k+1} = (D - L)^{-1} U x^{k} + (D - L)^{-1} b

    Parameters
    ----------
    A : array_like
        Coefficient matrix (n x n).
    b : array_like
        Right-hand side vector (n,).
    x0 : array_like
        Initial guess (n,).
    tolerance : float
        Tolerance for the stopping criterion (infinity norm of the change).
    niter : int
        Maximum number of iterations.

    Returns
    -------
    x : np.ndarray
        Last approximation of the solution vector.
    rho : float
        Spectral radius of the iteration matrix T.
    """
    # Print formatting
    np.set_printoptions(precision=6, suppress=True, floatmode='fixed')

    # Convert to column arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    x0 = np.array(x0, dtype=float).reshape(-1, 1)

    n = A.shape[0]

    # --------- PRE-CHECKS ---------

    # 1) Square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError("Error: matrix A must be square to apply Gauss-Seidel.")

    # 2) Compatible dimension for b
    if b.shape[0] != n:
        raise ValueError(
            f"Error: incompatible dimensions between A (n={n}) and b (m={b.shape[0]})."
        )

    # 3) Non-zero diagonal (required for (D-L)^{-1})
    diag = np.diagonal(A)
    if np.any(np.isclose(diag, 0.0)):
        idx = int(np.where(np.isclose(diag, 0.0))[0][0])
        raise ValueError(
            f"Error: Gauss-Seidel cannot be applied. "
            f"Matrix A has a zero diagonal element at row {idx+1} "
            f"(a_{idx+1},{idx+1} = 0)."
        )

    # --------- Decomposition A = D - L - U ---------
    D = np.diag(diag)
    L = -np.tril(A, -1)  # lower part with sign, so A = D - L - U
    U = -np.triu(A,  1)  # upper part with sign

    # Attempt to invert (D - L)
    try:
        inv_DL = np.linalg.inv(D - L)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Numerical error: matrix (D - L) is singular. "
            "Gauss-Seidel cannot be applied ((D-L)^{-1} does not exist)."
        )

    # Gauss-Seidel iteration matrix and vector
    T = inv_DL @ U
    C = inv_DL @ b

    # Spectral radius
    rho = spectral_radius(T)

    # --------- Print pre-analysis ---------
    print("Gauss-Seidel\n")
    print("Results:\n")

    print("T:")
    print(T)

    print("\nC:")
    print(C)

    print(f"\nspectral radius (rho(T_G)):\n{rho:.6f}\n")

    # Convergence analysis
    if np.isnan(rho):
        print("⚠️  Warning: could not compute the spectral radius of T (NaN).\n")
    elif rho >= 1.0:
        print(
            "⚠️  Warning: spectral radius ρ(T_G) is greater than or equal to 1.\n"
            "    Gauss-Seidel does NOT guarantee convergence for this system.\n"
        )
    else:
        print("✅ Theoretical convergence condition satisfied: ρ(T_G) < 1.\n")

    # Diagonal dominance (sufficient condition)
    if is_diagonally_dominant(A):
        print("✅ Matrix A is row-wise diagonally dominant.\n")
    else:
        print(
            "⚠️  Matrix A is NOT strictly diagonally dominant.\n"
            "    Gauss-Seidel may fail to converge, although this condition is not necessary.\n"
        )

    # --------- Gauss-Seidel iteration (matrix form) ---------
    iteration = 0
    error = 1.0  # initial error

    print("| iter |     E      | x components...")
    print("-" * 70)

    # Iteration 0
    x_str = "  ".join(f"{val:.6f}" for val in x0.flatten())
    print(f"|  {iteration:>3} | {'-':>10} | {x_str}")

    while error > tolerance and iteration < niter:
        # Matrix-form Gauss-Seidel update
        x1 = T @ x0 + C
        error = float(np.linalg.norm(x1 - x0, np.inf))

        x0 = x1
        iteration += 1

        x_str = "  ".join(f"{val:.6f}" for val in x0.flatten())
        print(f"| {iteration:>4} | {error:>10.1e} | {x_str}")

    print("-" * 70)
    if error <= tolerance:
        print(f"\nMatrix Gauss-Seidel: solution found in {iteration} iterations.")
        print("Approximate solution x:")
        for val in x0.flatten():
            print(f"{val:.6f}")
    else:
        print(f"\nMatrix Gauss-Seidel: failed after {niter} iterations.")
        print("Last approximation x:")
        for val in x0.flatten():
            print(f"{val:.6f}")

    return x0.flatten(), rho


# -------------------------------------------------------------------
# WRAPPERS for the site invoker (Django)
# -------------------------------------------------------------------
def _extract_extras(extras):
    """Takes extras from the site and assigns reasonable defaults."""
    if extras is None:
        extras = {}
    tol = float(extras.get("tol", 1e-7))
    nmax = int(extras.get("max_iter", 60))
    x0 = extras.get("x0", None)
    return tol, nmax, x0


def _coerce_vec(v, n):
    """Converts to a vector (n,) with zero-padding defaults."""
    if v is None:
        return np.zeros(n, dtype=float)
    arr = np.array(v, dtype=float).reshape(-1)
    if arr.size != n:
        # if dimension is wrong, we truncate/pad
        out = np.zeros(n, dtype=float)
        out[:min(n, arr.size)] = arr[:min(n, arr.size)]
        return out
    return arr


def run(A, b, extras=None):
    """Generic name detected by the site. DOES NOT print anything extra besides the method."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    tol, nmax, x0_in = _extract_extras(extras)
    x0 = _coerce_vec(x0_in, A.shape[0])
    # Call the core method, which prints the console:
    gauss_seidel_matricial(A, b, x0, tol, nmax)


# additional aliases in case the invoker searches for other names
def solve(A, b, extras=None):
    run(A, b, extras)


def algorithm(A, b, extras=None):
    run(A, b, extras)
