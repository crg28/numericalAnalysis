# methods/algorithms/jacobi.py
import numpy as np


def spectral_radius(T: np.ndarray) -> float:
    """
    Compute the spectral radius of T (maximum absolute eigenvalue).
    """
    try:
        eigenvalues = np.linalg.eigvals(T)
        return float(np.max(np.abs(eigenvalues)))
    except np.linalg.LinAlgError:
        return np.nan  # In case of convergence error


def jacobi_core(A, b, x0, tol, niter):
    """
    Core Jacobi implementation:
        A: coefficient matrix
        b: RHS vector
        x0: initial guess
        tol: tolerance
        niter: maximum number of iterations

    Prints the iteration table and returns (x, rho).
    """
    # 6 decimal places
    np.set_printoptions(precision=6, suppress=True, floatmode="fixed")

    iteration = 0
    error = 1.0  # initial error

    # Ensure numpy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    x0 = np.array(x0, dtype=float).reshape(-1, 1)

    # Decompose A into D, L, U
    D = np.diag(np.diagonal(A))
    L = -np.tril(A, -1)
    U = -np.triu(A,  1)

    # Inverse of D
    inv_D = np.diag(1.0 / np.diagonal(A))

    # Jacobi iteration matrix T and vector C
    T = inv_D @ (L + U)
    C = inv_D @ b

    # Spectral radius
    rho = spectral_radius(T)

    print("Jacobi\n")
    print("Results:\n")

    print("T:")
    print(T)

    print("\nC:")
    print(C)

    print(f"\nSpectral radius:\n{rho:.6f}\n")

    # Iteration table
    print("| iter |     E      | x components...")
    print("-" * 60)

    # Iteration 0
    x_str = "  ".join(f"{val:.6f}" for val in x0.flatten())
    print(f"|  {iteration:>3} | {'-':>10} | {x_str}")

    # Iteration loop
    while error > tol and iteration < niter:
        x1 = T @ x0 + C  # Jacobi update
        error = float(np.linalg.norm(x1 - x0, np.inf))

        x0 = x1
        iteration += 1

        x_str = "  ".join(f"{val:.6f}" for val in x0.flatten())
        print(f"| {iteration:>4} | {error:>10.1e} | {x_str}")

    print("-" * 60)
    if error < tol:
        print(f"\nJacobi method: solution found in {iteration} iterations.")
        print("Solution vector x:")
        for val in x0.flatten():
            print(f"{val:.6f}")
    else:
        print(f"\nJacobi method: failed to converge in {niter} iterations.")

    return x0.flatten(), rho


# ------------------------------------------------------
# Public entry point for Django
# ------------------------------------------------------
def jacobi(A, b, x0=None, tol=1e-7, max_iter=60):
    """
    Public Jacobi function used by the Django view.

    Parameters
    ----------
    A : array_like
        Coefficient matrix.
    b : array_like
        Right-hand side vector.
    x0 : array_like or None
        Initial guess. If None, uses zeros.
    tol : float
        Tolerance for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if x0 is None:
        x0 = np.zeros_like(b, dtype=float)
    else:
        x0 = np.array(x0, dtype=float)

    x, rho = jacobi_core(A, b, x0, tol, max_iter)

    # Optionally, we could print a short summary here (already printed inside core).
    return x, rho


# ------------------------------------------------------
# Legacy test (local run)
# ------------------------------------------------------
if __name__ == "__main__":
    A = np.array([
        [4,  -1,  0,  3],
        [1, 15.5, 3,  8],
        [0,  -1.3, -4, 1.1],
        [14,  5, -2, 30],
    ])
    b = np.array([1, 1, 1, 1])
    x0 = np.array([0, 0, 0, 0])

    jacobi(A, b, x0=x0, tol=1e-7, max_iter=60)
