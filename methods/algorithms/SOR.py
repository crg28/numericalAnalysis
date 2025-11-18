# methods/algorithms/SOR.py
import numpy as np


# ---------------------------------------------------------
# Utilities: spectral radius and SPD check
# ---------------------------------------------------------
def spectral_radius(T: np.ndarray) -> float:
    """Return max |eigenvalue| of T."""
    try:
        vals = np.linalg.eigvals(T)
        return float(np.max(np.abs(vals)))
    except np.linalg.LinAlgError:
        return np.nan


def is_spd(A: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Check whether A is symmetric positive definite (SPD):
      1. A ≈ A^T
      2. All eigenvalues > 0
    """
    A = np.array(A, dtype=float)
    if not np.allclose(A, A.T, atol=tol):
        return False
    try:
        # eigvalsh is more stable for symmetric matrices
        vals = np.linalg.eigvalsh((A + A.T) / 2.0)
        return np.all(vals > tol)
    except np.linalg.LinAlgError:
        return False


# ---------------------------------------------------------
# Build T_w and C_w
# ---------------------------------------------------------
def build_iteration_matrices(A, b, w):
    """
    Build x_{k+1} = T x_k + C for SOR with relaxation parameter w.

    A = D - L - U   (D diagonal, L strict lower, U strict upper)
    T = (D - wL)^{-1}[(1-w)D + wU]
    C = w (D - wL)^{-1} b
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    D = np.diag(np.diagonal(A))
    L = -np.tril(A, -1)
    U = -np.triu(A,  1)

    # May raise LinAlgError if (D - wL) is singular
    inv_term = np.linalg.inv(D - w * L)

    T = inv_term @ ((1.0 - w) * D + w * U)
    C = w * inv_term @ b
    return T, C


# ---------------------------------------------------------
# Core SOR Method
# ---------------------------------------------------------
def sor_run(A, b, x0, w, tol, nmax):
    """
    SOR using affine form x_{k+1} = T x_k + C.

    Guarantees / checks:
    - A must be square
    - No diagonal element can be zero (required for (D - wL)^{-1})
    - 0 < w < 2 (useful relaxation range)
    - Detects singularity of (D - wL)
    - Analyzes convergence via spectral radius ρ(T_w)
    - Indicates if A is SPD (sufficient condition together with 0 < w < 2)

    Returns:
      T, C, rho, history, converged
    """
    np.set_printoptions(precision=6, suppress=True, floatmode="fixed")

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    xk = np.array(x0, dtype=float).reshape(-1, 1)

    n = A.shape[0]

    # --------- PRE-CHECKS ---------

    if A.shape[0] != A.shape[1]:
        raise ValueError("Error: matrix A must be square to apply SOR.")

    if b.shape[0] != n:
        raise ValueError(
            f"Error: incompatible dimensions between A (n={n}) and b (m={b.shape[0]})."
        )

    diag = np.diagonal(A)
    if np.any(np.isclose(diag, 0.0)):
        idx = int(np.where(np.isclose(diag, 0.0))[0][0])
        raise ValueError(
            f"Error: SOR cannot be applied. "
            f"Matrix A has a zero diagonal element at row {idx+1} "
            f"(a_{idx+1},{idx+1} = 0)."
        )

    if not (0.0 < w < 2.0):
        raise ValueError(
            f"Error: relaxation parameter w={w} is outside the useful range "
            "(0 < w < 2). SOR generally fails to converge otherwise."
        )

    # Build T and C, controlling singularity of (D - wL)
    try:
        T, C = build_iteration_matrices(A, b, w)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Numerical error: matrix (D - wL) is singular. "
            "SOR cannot be applied because (D - wL)^{-1} does not exist."
        )

    # Spectral radius
    rho = spectral_radius(T)

    # --------- Pre-iteration output ---------
    print("\nSOR (relaxation)\n")
    print("Results:\n")

    print("T:")
    for row in T:
        print("  " + "  ".join(f"{val: .6f}" for val in row))
    print()

    print("C:")
    print("  " + "  ".join(f"{val: .6f}" for val in C.flatten()))
    print()

    print("Spectral radius (rho(T_w)):")
    print(f"  {rho:.6f}\n")

    if np.isnan(rho):
        print("⚠️  Warning: spectral radius could not be computed (NaN).\n")
    elif rho >= 1.0:
        print(
            "⚠️  Warning: spectral radius ρ(T_w) is greater than or equal to 1.\n"
            "    SOR does NOT guarantee convergence in this case.\n"
        )
    else:
        print("✅ Theoretical convergence condition satisfied: ρ(T_w) < 1.\n")

    # SPD condition
    if is_spd(A) and (0.0 < w < 2.0):
        print(
            "✅ A is symmetric positive definite and 0 < w < 2.\n"
            "   Under these conditions, SOR convergence is guaranteed.\n"
        )
    else:
        print(
            "ℹ️  Note: SOR is also guaranteed to converge if A is SPD and 0 < w < 2.\n"
        )

    # --------- SOR Iteration ---------
    history = []
    converged = False

    # Iteration 0
    history.append({"iter": 0, "err": None, "x": xk.flatten().copy()})

    for k in range(1, nmax + 1):
        x_next = T @ xk + C
        err = float(np.linalg.norm(x_next - xk, ord=np.inf))

        history.append({"iter": k, "err": err, "x": x_next.flatten().copy()})

        xk = x_next
        if err < tol:
            converged = True
            break

    return T, C, rho, history, converged


# ---------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------
def pretty_print_results(T, C, rho, history, tol, converged):
    print("| {:>4s} | {:>8s} | {:>20s} |".format("iter", "E", "x (components...)"))
    for entry in history:
        it = entry["iter"]
        err = entry["err"]
        xvals = entry["x"]
        err_str = "   -" if err is None else f"{err:.1e}"
        x_str = " ".join(f"{xi:.6f}" for xi in xvals)
        print(f"| {it:>4d} | {err_str:>8s} | {x_str}")
    print()

    print("Converged:", converged)
    print(f"Stopped with Tol = {tol}")
    print(f"Iterations recorded: {history[-1]['iter']}")


# ---------------------------------------------------------
# High-level wrapper for Django / console
# ---------------------------------------------------------
def sor(A, b, x0=None, w=1.0, tol=1e-7, max_iter=100):
    """
    High-level SOR wrapper used by the Django view.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if x0 is None:
        x0 = np.zeros_like(b, dtype=float)
    else:
        x0 = np.array(x0, dtype=float)

    T, C, rho, history, converged = sor_run(A, b, x0, w, tol, max_iter)
    pretty_print_results(T, C, rho, history, tol, converged)


# -------- Example test --------
if __name__ == "__main__":
    A = np.array([
        [4,   -1,   0,   3],
        [1,   15.5, 3,   8],
        [0,   -1.3, -4,  1.1],
        [14,   5,   -2, 30]
    ], dtype=float)

    b = np.array([1, 1, 1, 1], dtype=float)

    x0   = np.zeros(4)
    tol  = 1e-7
    nmax = 100
    w    = 1.5

    T, C, rho, history, converged = sor_run(A, b, x0, w, tol, nmax)
    pretty_print_results(T, C, rho, history, tol, converged)
