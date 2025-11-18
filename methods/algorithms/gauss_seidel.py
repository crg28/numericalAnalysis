import numpy as np

# -------------------------------------------------------------------
# Utilidades: radio espectral y dominancia diagonal
# -------------------------------------------------------------------
def radio_espectral(T: np.ndarray) -> float:
    """
    Calcula el radio espectral de una matriz T (máx |eigenvalor|).
    """
    try:
        eigenvalores = np.linalg.eigvals(T)
        return float(np.max(np.abs(eigenvalores)))
    except np.linalg.LinAlgError:
        return np.nan  # En caso de error numérico


def es_dominante_diagonal(A: np.ndarray) -> bool:
    """
    Verifica dominancia diagonal por filas:
        |a_ii| >= sum_{j != i} |a_ij|  para todas las filas i.
    (Condición suficiente de convergencia para Jacobi / Gauss-Seidel.)
    """
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        suma_no_diag = np.sum(np.abs(A[i, :])) - diag
        if diag < suma_no_diag:
            return False
    return True


# -------------------------------------------------------------------
# Núcleo del método de Gauss-Seidel (forma matricial)
# -------------------------------------------------------------------
def gauss_seidel_matricial(A, b, x0, tolerancia, niter):
    """
    Implementación matricial del método de Gauss-Seidel:

        A x = b
        A = D - L - U
        (D - L) x^{k+1} = U x^{k} + b
        x^{k+1} = (D - L)^{-1} U x^{k} + (D - L)^{-1} b

    Parámetros
    ----------
    A : array_like
        Matriz de coeficientes (n x n).
    b : array_like
        Vector de términos independientes (n,).
    x0 : array_like
        Aproximación inicial (n,).
    tolerancia : float
        Tolerancia para el criterio de parada (norma infinito del cambio).
    niter : int
        Máximo número de iteraciones.

    Retorna
    -------
    x : np.ndarray
        Última aproximación del vector solución.
    ro : float
        Radio espectral de la matriz de iteración T.
    """
    # Formato de impresión
    np.set_printoptions(precision=6, suppress=True, floatmode='fixed')

    # Convertir a arreglos columna
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    x0 = np.array(x0, dtype=float).reshape(-1, 1)

    n = A.shape[0]

    # --------- COMPROBACIONES PREVIAS ---------

    # 1) Matriz cuadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("Error: la matriz A debe ser cuadrada para aplicar Gauss-Seidel.")

    # 2) Dimensión de b compatible
    if b.shape[0] != n:
        raise ValueError(
            f"Error: dimensiones incompatibles entre A (n={n}) y b (m={b.shape[0]})."
        )

    # 3) Diagonal no nula (requisito para (D-L)^{-1})
    diag = np.diagonal(A)
    if np.any(np.isclose(diag, 0.0)):
        idx = int(np.where(np.isclose(diag, 0.0))[0][0])
        raise ValueError(
            f"Error: Gauss-Seidel no se puede aplicar. "
            f"La matriz A tiene un elemento diagonal nulo en la fila {idx+1} "
            f"(a_{idx+1},{idx+1} = 0)."
        )

    # --------- Descomposición A = D - L - U ---------
    D = np.diag(diag)
    L = -np.tril(A, -1)  # parte inferior, con signo, de forma que A = D - L - U
    U = -np.triu(A,  1)  # parte superior, con signo

    # Intentar invertir (D - L)
    try:
        inv_DL = np.linalg.inv(D - L)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Error numérico: La matriz (D - L) es singular. "
            "No se puede aplicar Gauss-Seidel (no existe (D-L)^{-1})."
        )

    # Matriz de iteración y vector C para Gauss-Seidel
    T = inv_DL @ U
    C = inv_DL @ b

    # Radio espectral
    ro = radio_espectral(T)

    # --------- Impresión de información previa ---------
    print("Gauss-Seidel\n")
    print("Resultados:\n")

    print("T:")
    print(T)

    print("\nC:")
    print(C)

    print(f"\nradio espectral (rho(T_G)):\n{ro:.6f}\n")

    # Análisis de convergencia
    if np.isnan(ro):
        print("⚠️  Advertencia: no se pudo calcular el radio espectral de T (NaN).\n")
    elif ro >= 1.0:
        print(
            "⚠️  Advertencia: el radio espectral ρ(T_G) es mayor o igual que 1.\n"
            "    El método de Gauss-Seidel no garantiza convergencia para este sistema.\n"
        )
    else:
        print("✅ Condición teórica de convergencia cumplida: ρ(T_G) < 1.\n")

    # Dominancia diagonal (condición suficiente)
    if es_dominante_diagonal(A):
        print("✅ La matriz A es diagonalmente dominante (fila a fila).\n")
    else:
        print(
            "⚠️  La matriz A no es estrictamente diagonalmente dominante.\n"
            "    Gauss-Seidel podría no converger, aunque esta condición no es necesaria.\n"
        )

    # --------- Iteración de Gauss-Seidel (forma matricial) ---------
    iteracion = 0
    error = 1.0  # error inicial

    print("| iter |     E      | x components...")
    print("-" * 70)

    # Iteración 0
    x_str = "  ".join(f"{val:.6f}" for val in x0.flatten())
    print(f"|  {iteracion:>3} | {'-':>10} | {x_str}")

    while error > tolerancia and iteracion < niter:
        # Actualización Gauss-Seidel en forma matricial
        x1 = T @ x0 + C
        error = float(np.linalg.norm(x1 - x0, np.inf))

        x0 = x1
        iteracion += 1

        x_str = "  ".join(f"{val:.6f}" for val in x0.flatten())
        print(f"| {iteracion:>4} | {error:>10.1e} | {x_str}")

    print("-" * 70)
    if error <= tolerancia:
        print(f"\nGauss-Seidel matricial: solución encontrada en {iteracion} iteraciones.")
        print("Solución aproximada x:")
        for val in x0.flatten():
            print(f"{val:.6f}")
    else:
        print(f"\nGauss-Seidel matricial: fracasó en {niter} iteraciones.")
        print("Última aproximación x:")
        for val in x0.flatten():
            print(f"{val:.6f}")

    return x0.flatten(), ro


# -------------------------------------------------------------------
# WRAPPERS para el invocador del sitio (Django)
# -------------------------------------------------------------------
def _extract_extras(extras):
    """Toma extras del sitio y pone defaults razonables."""
    if extras is None:
        extras = {}
    tol = float(extras.get("tol", 1e-7))
    nmax = int(extras.get("max_iter", 60))
    x0 = extras.get("x0", None)
    return tol, nmax, x0


def _coerce_vec(v, n):
    """Convierte a vector (n,) con ceros por defecto."""
    if v is None:
        return np.zeros(n, dtype=float)
    arr = np.array(v, dtype=float).reshape(-1)
    if arr.size != n:
        # si vino mal dimensionado, truncamos/rellenamos
        out = np.zeros(n, dtype=float)
        out[:min(n, arr.size)] = arr[:min(n, arr.size)]
        return out
    return arr


def run(A, b, extras=None):
    """Nombre genérico que el sitio detecta. NO imprime nada extra aparte del método."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    tol, nmax, x0_in = _extract_extras(extras)
    x0 = _coerce_vec(x0_in, A.shape[0])
    # Llama al núcleo, que imprime toda la consola:
    gauss_seidel_matricial(A, b, x0, tol, nmax)


# aliases adicionales por si el invocador buscara otros nombres
def solve(A, b, extras=None):
    run(A, b, extras)


def algorithm(A, b, extras=None):
    run(A, b, extras)
