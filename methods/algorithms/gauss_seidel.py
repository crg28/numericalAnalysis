import numpy as np

# -------------------------------------------------------------------
# Tu método, sin cambios
# -------------------------------------------------------------------
def radio_espectral(T):
    """
    Calcula el radio espectral de una matriz T (máx |eigenvalor|).
    """
    try:
        eigenvalores = np.linalg.eigvals(T)
        return np.max(np.abs(eigenvalores))
    except np.linalg.LinAlgError:
        return np.nan  # En caso de error numérico

def gauss_seidel_matricial(A, b, x0, tolerancia, niter):
    # 6 decimales
    np.set_printoptions(precision=6, suppress=True, floatmode='fixed')

    iteracion = 0
    error = 1  # Error inicial

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).T
    x0 = np.array(x0, dtype=float).T

    # Descomposición de la matriz A en D, L y U
    D = np.diag(np.diagonal(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)

    # Inversa de (D - L)
    inv_DL = np.linalg.inv(D - L)

    # Matriz de transición y vector C (Gauss-Seidel)
    T = inv_DL @ U
    C = inv_DL @ b

    # Radio espectral
    ro = radio_espectral(T)

    print("Gauss-Seidel\n")
    print("Resultados:\n")
    print("T:")
    print(T)

    print("\nC:")
    print(C)

    print(f"\nradio espectral:\n{ro:.6f}\n")

    # Tabla de iteraciones
    print("| iter |     E      | ")
    print("-" * 50)

    x_str = "  ".join([f"{val:.6f}" for val in x0])
    print(f"|  {iteracion: >3} | {'-': >10} | {x_str}")

    while error > tolerancia and iteracion < niter:
        x1 = T @ x0 + C  # Fórmula de Gauss-Seidel
        error = np.linalg.norm(x1 - x0, np.inf)
        x0 = x1
        iteracion += 1
        x_str = "  ".join([f"{val:.6f}" for val in x0])
        print(f"| {iteracion: >4} | {error: >10.1e} | {x_str}")

    print("-" * 50)
    if error < tolerancia:
        print(f"\nGauss-Seidel matricial: Solución encontrada en {iteracion} iteraciones.")
        print(f"Solución: {x0}")
    else:
        print(f"\nGauss-Seidel matricial: Fracasó en {niter} iteraciones.")

    return x0, ro

# -------------------------------------------------------------------
# WRAPPERS para el invocador del sitio:
# - El capturador de stdout busca nombres tipo: "run", "solve", "algorithm", etc.
# - Estos adaptan los datos que vienen del formulario y llaman a tu función.
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
    """Nombre genérico que el sitio detecta. NO imprime nada extra."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    tol, nmax, x0_in = _extract_extras(extras)
    x0 = _coerce_vec(x0_in, A.shape[0])
    # Llama a tu función, que imprime todo el bloque consola:
    gauss_seidel_matricial(A, b, x0, tol, nmax)

# aliases adicionales por si el invocador buscara otros nombres
def solve(A, b, extras=None):     run(A, b, extras)
def algorithm(A, b, extras=None): run(A, b, extras)

# Nada de "if __name__ == '__main__':" para que NO imprima al importar.
