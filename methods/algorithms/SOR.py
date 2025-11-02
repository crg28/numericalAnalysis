import numpy as np

def spectral_radius(T: np.ndarray) -> float:
    """
    Computes the spectral radius of T = max |lambda_i|.
    """
    vals, _ = np.linalg.eig(T)
    return float(np.max(np.abs(vals)))


def build_iteration_matrices(A, b, w):
    """
    From A, b, and relaxation factor w, build:
    - T: iteration matrix
    - C: constant vector
    such that x_{k+1} = T x_k + C
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    D = np.diag(np.diagonal(A))        # diagonal
    L = -np.tril(A, -1)                # strict lower (with minus sign)
    U = -np.triu(A,  1)                # strict upper (with minus sign)

    inv_term = np.linalg.inv(D - w * L)
    T = inv_term @ ((1.0 - w) * D + w * U)
    C = w * inv_term @ b

    return T, C


def sor_run(A, b, x0, w, tol, nmax):
    """
    Runs SOR using the affine form x_{k+1} = T x_k + C.

    Stopping rule:
    error = ||x_{k+1} - x_k||_inf <= tol   OR   k == nmax

    Returns:
    - T, C, rho
    - history: list of dicts for each iteration:
        {
          "iter": k,
          "err": err (None for k=0),
          "x":   current approximation as 1D array
        }
    - converged: bool
    """
    # construct iteration matrices
    T, C = build_iteration_matrices(A, b, w)

    # initial guess
    xk = np.array(x0, dtype=float).reshape(-1, 1)

    history = []
    # iteration 0 entry
    history.append({
        "iter": 0,
        "err": None,
        "x": xk.flatten().copy()
    })

    k = 0
    converged = False
    err = np.inf

    while k < nmax and err > tol:
        x_next = T @ xk + C
        diff = x_next - xk

        # use infinity norm (max abs component) like your screenshot table's E
        err = float(np.linalg.norm(diff, ord=np.inf))

        k += 1
        history.append({
            "iter": k,
            "err": err,
            "x": x_next.flatten().copy()
        })

        xk = x_next
        if err <= tol:
            converged = True
            break

    rho = spectral_radius(T)

    return T, C, rho, history, converged


def pretty_print_results(T, C, rho, history):
    """
    Prints output in the same style you showed:
    - "SOR (relajación)"
    - T:
    - C:
    - spectral radius:
    - iteration table
    """
    print("\nSOR (relaxation)\n")
    print("Results:\n")

    # Print T
    print("T:")
    for row in T:
        print("  " + "  ".join(f"{val: .6f}" for val in row))
    print()

    # Print C
    print("C:")
    print("  " + "  ".join(f"{val: .6f}" for val in C.flatten()))
    print()

    # Print spectral radius
    print("spectral radius:")
    print(f"  {rho:.6f}")
    print()

    # Build table header
    # Your screenshot showed something like:
    # | iter |  E    |     x-values...
    # We'll align similar formatting.
    # Width tuning so columns line up decently.
    print("| {:>4s} | {:>8s} | {:>12s} |".format("iter", "E", "x (components...)"))
    for entry in history:
        it = entry["iter"]
        err = entry["err"]
        xvals = entry["x"]

        if err is None:
            err_str = "   -"
        else:
            # scientific-ish formatting like 6.2e-01
            err_str = f"{err:.1e}"

        # join x components with spaces
        x_str = " ".join(f"{xi:.6f}" for xi in xvals)

        print(f"| {it:>4d} | {err_str:>8s} | {x_str}")


def read_user_input():
    """
    Ask ONLY for:
    - n
    - A
    - b
    - x0
    - Tol
    - Nmax
    w is fixed to 1.5 (as per requirement).
    """
    print("SOR input: A, b, x0, Tol, w = 1.5, Nmax.\n")

    n = int(input("Enter n (matrix A is n x n): "))

    print("\nEnter A row by row (space separated):")
    A = []
    for i in range(n):
        row = [float(v) for v in input(f"A row {i+1}: ").split()]
        if len(row) != n:
            raise ValueError(f"Row {i+1} must have {n} values.")
        A.append(row)

    print("\nEnter b (space separated, length n):")
    b = [float(v) for v in input("b: ").split()]
    if len(b) != n:
        raise ValueError("b must have n values.")

    print("\nEnter x0 (space separated, length n):")
    x0 = [float(v) for v in input("x0: ").split()]
    if len(x0) != n:
        raise ValueError("x0 must have n values.")

    tol = float(input("\nEnter tolerance Tol (example 1e-7): "))
    nmax = int(input("Enter Nmax (max iterations): "))

    w = 1.5
    print(f"\nUsing w = {w} (fixed).")

    return A, b, x0, tol, nmax, w


if __name__ == "__main__":
    # 1. Read user data
    A, b, x0, tol, nmax, w = read_user_input()

    # 2. Run SOR
    T, C, rho, history, converged = sor_run(
        A=A,
        b=b,
        x0=x0,
        w=w,
        tol=tol,
        nmax=nmax
    )

    # 3. Print formatted results
    pretty_print_results(T, C, rho, history)

    # 4. (Optional) final note
    print("\nConverged:", converged)
    print(f"Stopped with Tol = {tol}")
    print(f"Iterations performed: {history[-1]['iter']}")
