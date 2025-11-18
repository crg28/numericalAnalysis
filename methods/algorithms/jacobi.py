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


def is_diagonally_dominant(A: np.ndarray) -> bool:
    """
    Check (row-wise) diagonal dominance:
        |a_ii| >= sum_{j != i} |a_ij|  for all i
    """
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag_sum = np.sum(np.abs(A[i, :])) - diag
        if diag < off_diag_sum:
            return False
    return True


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

    # Ensure numpy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    x0 = np.array(x0, dtype=float).reshape(-1, 1)

    n = A.shape[0]

    # --------- COMPROBACIONES PREVIAS ---------

    # 1) Verificar que A sea cuadrada y dimensions compatibles con b
    if A.shape[0] != A.shape[1]:
        raise ValueError("Error: la matriz A debe ser cuadrada para aplicar Jacobi.")

    if b.shape[0] != n:
        raise ValueError(
            f"Error: dimensiones incompatibles entre A (n={n}) y b (m={b.shape[0]})."
        )

    # 2) Comprobar que ningún elemento diagonal sea cero
    diag = np.diagonal(A)
    if np.any(np.isclose(diag, 0.0)):
        idx = int(np.where(np.isclose(diag, 0.0))[0][0])
        raise ValueError(
            f"Error: Jacobi no se puede aplicar. "
            f"La matriz A tiene un elemento diagonal nulo en la fila {idx+1} (a_{idx+1},{idx+1} = 0)."
        )

    # --------- Descomposición A = D - L - U ---------

    D = np.diag(diag)
    L = -np.tril(A, -1)  # parte inferior con signo, consistente con A = D - L - U
    U = -np.triu(A,  1)  # parte superior con signo

    # Inverse of D (diagonal matrix)
    inv_D = np.diag(1.0 / diag)

    # Jacobi iteration matrix T and vector C
    T = inv_D @ (L + U)
    C = inv_D @ b

    # Spectral radius
    rho = spectral_radius(T)

    # --------- Impresión de resultados previos ---------
    print("Jacobi\n")
    print("Results:\n")

    print("T:")
    print(T)

    print("\nC:")
    print(C)

    print(f"\nSpectral radius (rho):\n{rho:.6f}\n")

    # Comprobación de convergencia teórica (radio espectral)
    if np.isnan(rho):
        print("Advertencia: no se pudo calcular el radio espectral de T (NaN).")
    elif rho >= 1.0:
        print(
            "⚠️  Advertencia: el radio espectral ρ(T) es mayor o igual que 1.\n"
            "   El método de Jacobi no garantiza convergencia para este sistema.\n"
        )
    else:
        print(
            "✅ Condición teórica de convergencia cumplida: ρ(T) < 1.\n"
        )

    # Comprobación simple de dominancia diagonal
    if is_diagonally_dominant(A):
        print("✅ La matriz A es diagonalmente dominante (fila a fila).\n")
    else:
        print(
            "⚠️  La matriz A no es estrictamente diagonalmente dominante.\n"
            "   Jacobi podría no converger, aunque esta condición no es necesaria.\n"
        )

    # --------- Iteración de Jacobi ---------
    iteration = 0
    error = 1.0  # initial error

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
    if error <= tol:
        print(f"\nJacobi method: solution found in {iteration} iterations.")
        print("Solution vector x:")
        for val in x0.flatten():
            print(f"{val:.6f}")
    else:
        print(f"\nJacobi method: failed to converge in {niter} iterations.")
        print("Último vector aproximado x:")
        for val in x0.flatten():
            print(f"{val:.6f}")

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
