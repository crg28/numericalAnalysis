# methods/algorithms/SOR.py
import numpy as np


# ---------------------------------------------------------
# Utilidades: radio espectral y SPD
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
    Comprueba si A es simétrica definida positiva:
      1. A ≈ A^T
      2. Todos los autovalores > 0
    """
    A = np.array(A, dtype=float)
    if not np.allclose(A, A.T, atol=tol):
        return False
    try:
        # Para matrices simétricas es más estable eigvalsh
        vals = np.linalg.eigvalsh((A + A.T) / 2.0)
        return np.all(vals > tol)
    except np.linalg.LinAlgError:
        return False


# ---------------------------------------------------------
# Construcción de T_w y C_w
# ---------------------------------------------------------
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

    # Puede lanzar LinAlgError si (D - wL) es singular
    inv_term = np.linalg.inv(D - w * L)

    T = inv_term @ ((1.0 - w) * D + w * U)
    C = w * inv_term @ b
    return T, C


# ---------------------------------------------------------
# Núcleo del método SOR
# ---------------------------------------------------------
def sor_run(A, b, x0, w, tol, nmax):
    """
    SOR via affine form x_{k+1} = T x_k + C.

    Garantías / comprobaciones:
    - Verifica que A sea cuadrada.
    - Verifica que ningún a_ii sea cero (requisito para (D - wL)^(-1)).
    - Verifica que 0 < w < 2 (rango útil de relajación).
    - Detecta si (D - wL) es singular.
    - Analiza convergencia via radio espectral ρ(T_w).
    - Indica si A es SPD (condición suficiente de convergencia junto con 0 < w < 2).

    Retorna:
      T, C, rho, history, converged
    """
    np.set_printoptions(precision=6, suppress=True, floatmode="fixed")

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    xk = np.array(x0, dtype=float).reshape(-1, 1)

    n = A.shape[0]

    # --------- COMPROBACIONES PREVIAS ---------

    # A cuadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("Error: la matriz A debe ser cuadrada para aplicar SOR.")

    # dim de b
    if b.shape[0] != n:
        raise ValueError(
            f"Error: dimensiones incompatibles entre A (n={n}) y b (m={b.shape[0]})."
        )

    # Diagonal no nula
    diag = np.diagonal(A)
    if np.any(np.isclose(diag, 0.0)):
        idx = int(np.where(np.isclose(diag, 0.0))[0][0])
        raise ValueError(
            f"Error: SOR no se puede aplicar. "
            f"La matriz A tiene un elemento diagonal nulo en la fila {idx+1} "
            f"(a_{idx+1},{idx+1} = 0)."
        )

    # Parámetro w en rango útil
    if not (0.0 < w < 2.0):
        raise ValueError(
            f"Error: el parámetro de relajación w={w} está fuera del rango útil "
            "(0 < w < 2). El método SOR generalmente no converge en este caso."
        )

    # Construir T y C, controlando singularidad de (D - wL)
    try:
        T, C = build_iteration_matrices(A, b, w)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Error numérico: la matriz (D - wL) es singular. "
            "No se puede aplicar SOR porque (D - wL)^{-1} no existe."
        )

    # Radio espectral
    rho = spectral_radius(T)

    # --------- Impresión de info previa ---------
    print("\nSOR (relaxation)\n")
    print("Results:\n")

    print("T:")
    for row in T:
        print("  " + "  ".join(f"{val: .6f}" for val in row))
    print()

    print("C:")
    print("  " + "  ".join(f"{val: .6f}" for val in C.flatten()))
    print()

    print("spectral radius (rho(T_w)):")
    print(f"  {rho:.6f}\n")

    # Análisis de convergencia
    if np.isnan(rho):
        print("⚠️  Advertencia: no se pudo calcular el radio espectral (NaN).\n")
    elif rho >= 1.0:
        print(
            "⚠️  Advertencia: el radio espectral ρ(T_w) es mayor o igual que 1.\n"
            "    El método SOR no garantiza convergencia para este sistema.\n"
        )
    else:
        print("✅ Condición teórica de convergencia cumplida: ρ(T_w) < 1.\n")

    # Condición suficiente: A SPD y 0 < w < 2
    if is_spd(A) and (0.0 < w < 2.0):
        print(
            "✅ A es simétrica definida positiva y 0 < w < 2.\n"
            "   Bajo estas condiciones, la convergencia del método SOR está garantizada.\n"
        )
    else:
        print(
            "ℹ️  Nota: la convergencia de SOR también está garantizada si A es SPD y 0 < w < 2.\n"
        )

    # --------- Iteración SOR usando la forma matricial ---------
    history = []
    converged = False

    # Iteración 0
    history.append({"iter": 0, "err": None, "x": xk.flatten().copy()})
    error = 1.0

    # Iteraciones hasta nmax, registrando siempre la iteración donde se cumple tol
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
# Impresión “bonita” de los resultados
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
# Wrapper de alto nivel para Django / consola
# ---------------------------------------------------------
def sor(A, b, x0=None, w=1.0, tol=1e-7, max_iter=100):
    """
    High-level SOR wrapper used by the Django view.

    Parameters
    ----------
    A : array_like
        Coefficient matrix.
    b : array_like
        Right-hand-side vector.
    x0 : array_like or None
        Initial guess. If None, uses zeros.
    w : float
        Relaxation factor (0 < w < 2).
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

    T, C, rho, history, converged = sor_run(A, b, x0, w, tol, max_iter)

    pretty_print_results(T, C, rho, history, tol, converged)


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
    pretty_print_results(T, C, rho, history, tol, converged)
