# sor.py
import numpy as np

def spectral_radius(T: np.ndarray) -> float:
    """Return max |eigenvalue| of T."""
    vals, _ = np.linalg.eig(T)
    return float(np.max(np.abs(vals)))

def build_iteration_matrices(A, b, w):
    """
    Build x_{k+1} = T x_k + C for SOR with relaxation w.

    A = D - L - U   (D diagonal, L strict lower, U strict upper)
    T = (D - wL)^{-1}[(1-w)D + wU]
    C = w (D - wL)^{-1} b
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    D = np.diag(np.diagonal(A))
    L = -np.tril(A, -1)
    U = -np.triu(A,  1)

    inv_term = np.linalg.inv(D - w * L)
    T = inv_term @ ((1.0 - w) * D + w * U)
    C = w * inv_term @ b
    return T, C

def sor_run(A, b, x0, w, tol, nmax):
    """
    SOR via affine form x_{k+1} = T x_k + C.

    We always compute the next iterate, record it, then stop
    *after* recording the first iteration where error < tol.
    This avoids stopping one step early when err hits tol exactly.
    """
    T, C = build_iteration_matrices(A, b, w)

    xk = np.array(x0, dtype=float).reshape(-1, 1)

    history = [{"iter": 0, "err": None, "x": xk.flatten().copy()}]
    converged = False

    # Take up to nmax iterations, stopping after we *record* the first < tol
    for k in range(1, nmax + 1):
        x_next = T @ xk + C
        err = float(np.linalg.norm(x_next - xk, ord=np.inf))

        history.append({"iter": k, "err": err, "x": x_next.flatten().copy()})

        xk = x_next
        if err < tol:
            converged = True
            break

    rho = spectral_radius(T)
    return T, C, rho, history, converged

def pretty_print_results(T, C, rho, history):
    print("\nSOR (relaxation)\n")
    print("Results:\n")

    print("T:")
    for row in T:
        print("  " + "  ".join(f"{val: .6f}" for val in row))
    print()

    print("C:")
    print("  " + "  ".join(f"{val: .6f}" for val in C.flatten()))
    print()

    print("spectral radius:")
    print(f"  {rho:.6f}\n")

    print("| {:>4s} | {:>8s} | {:>12s} |".format("iter", "E", "x (components...)"))
    for entry in history:
        it = entry["iter"]
        err = entry["err"]
        xvals = entry["x"]
        err_str = "   -" if err is None else f"{err:.1e}"
        x_str = " ".join(f"{xi:.6f}" for xi in xvals)
        print(f"| {it:>4d} | {err_str:>8s} | {x_str}")

# -------- Fixed data (same as the other methods) --------
if __name__ == "__main__":
    A = np.array([
        [4,   -1,   0,   3],
        [1,   15.5, 3,   8],
        [0,   -1.3, -4,  1.1],
        [14,   5,   -2, 30]
    ], dtype=float)

    b = np.array([1, 1, 1, 1], dtype=float)

    x0   = np.zeros(4)   # (0, 0, 0, 0)
    tol  = 1e-7
    nmax = 100
    w    = 1.5

    T, C, rho, history, converged = sor_run(A, b, x0, w, tol, nmax)
    pretty_print_results(T, C, rho, history)

    print("\nConverged:", converged)
    print(f"Stopped with Tol = {tol}")
    print(f"Iterations recorded: {history[-1]['iter']}")
