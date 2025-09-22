import numpy as np
from fractions import Fraction

# ===================== Utilidades =====================

def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=L.dtype)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def back_substitution(U, y, tol=1e-12):
    n = U.shape[0]
    x = np.zeros_like(y, dtype=U.dtype)
    for i in range(n - 1, -1, -1):
        piv = U[i, i]
        if abs(float(piv)) < tol:
            raise np.linalg.LinAlgError("Pivote ~0 durante back_substitution (parcial).")
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / piv
    return x

def _matrix_lines(M, decimals=6, col_width=12):
    """Devuelve líneas de texto bonitas para imprimir una matriz."""
    fmt = "{:>" + str(col_width) + "." + str(decimals) + "f}"
    lines = []
    for i in range(M.shape[0]):
        row = " ".join(fmt.format(float(v)) for v in M[i])
        lines.append("[" + row + "]")
    return lines

def _boxed_print(title, sections, pad=1, min_width=72):
    """
    sections: lista de tuplas (heading:str, lines:list[str])
    Construye una caja con título y secciones separadas por espacios.
    """
    all_lines = []
    # Para cálculo de ancho: incluir headings y líneas
    for heading, lines in sections:
        all_lines.append("▸ " + heading)
        all_lines.extend(lines if lines else ["(vacío)"])
        all_lines.append("")  # separador visual

    if all_lines and all_lines[-1] == "":
        all_lines.pop()

    content_max = max([len(title)] + [len(s) for s in all_lines]) if all_lines else len(title)
    width = max(min_width, content_max + 4)

    top = "╭" + "─" * (width - 2) + "╮"
    sep = "├" + "─" * (width - 2) + "┤"
    bot = "╰" + "─" * (width - 2) + "╯"

    def pl(line): print("│ " + line.ljust(width - 3) + "│")

    print(top)
    pl((" " + title + " ").center(width - 3, "─"))
    print(sep)

    for heading, lines in sections:
        for _ in range(pad): pl("")
        pl("▸ " + heading)
        for _ in range(pad // 2): pl("")
        if not lines:
            pl("(vacío)")
        else:
            for line in lines:
                pl(line)
        for _ in range(pad): pl("")
    print(bot)

# ===================== Pivoteo Parcial =====================

def solve_partial_pivot(A, b, tol=1e-12):
    """
    Factoriza con pivoteo parcial (solo filas):  P A = L U
    Devuelve x, U, P  (no mostramos L aquí para mantener la impresión simple).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("A debe ser cuadrada (parcial).")
    if b.shape[0] != n:
        raise ValueError("Dimensión de b incompatible (parcial).")

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        # Selección de pivote máximo en columna k
        pivot_row = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[pivot_row, k]) < tol:
            raise np.linalg.LinAlgError(f"Matriz singular o pivote ~0 en columna {k} (parcial).")

        # Intercambio de filas
        if pivot_row != k:
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # Eliminación
        piv = U[k, k]
        for i in range(k + 1, n):
            m = U[i, k] / piv
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]

    # Resolver: P A x = L U x = P b
    Pb = P @ b
    y = forward_substitution(L, Pb)
    x = back_substitution(U, y, tol=tol)
    return x, U, P

def print_solution(title, A0, b, x, U, P, decimals=6, show_fractions=True, col_width=12):
    residuo = float(np.linalg.norm(A0 @ x - b, 2))
    condA   = float(np.linalg.cond(A0, 2))

    # Sección: Solución
    sol_lines = []
    for i, val in enumerate(x, start=1):
        base = f"x{i} = {float(val):.{decimals}f}"
        if show_fractions:
            base += f"   ({Fraction(float(val)).limit_denominator()})"
        sol_lines.append(base)

    # Sección: Métricas
    metrics_lines = [
        f"‖A x − b‖₂ = {residuo:.3e}",
        f"cond₂(A)   = {condA:.3e}",
        f"dtype      = {A0.dtype}",
    ]

    # Sección: PA (matriz tras pivoteo de filas)
    PA = P @ A0
    pa_lines = _matrix_lines(PA, decimals=decimals, col_width=col_width)

    # Sección: U
    u_lines = _matrix_lines(U, decimals=decimals, col_width=col_width)

    sections = [
        ("Solución", sol_lines),
        ("Métricas", metrics_lines),
        ("Matriz tras pivoteo (PA)", pa_lines),
        ("Matriz U (triangular superior)", u_lines),
    ]
    _boxed_print(title, sections, pad=1, min_width=72)

# ===================== Main =====================

if __name__ == "__main__":
    n = int(input("Ingrese el tamaño de la matriz A (n): "))
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    print("\nIngrese la matriz A fila por fila:")
    for i in range(n):
        fila = input(f"Fila {i+1} (separar con espacios): ").split()
        if len(fila) != n:
            raise ValueError("Número de columnas incorrecto.")
        A[i, :] = [float(num) for num in fila]

    print("\nIngrese el vector b:")
    for i in range(n):
        b[i] = float(input(f"b[{i+1}] = "))

    A0 = A.copy()
    x, U, P = solve_partial_pivot(A, b)
    print_solution("Pivoteo Parcial", A0, b, x, U, P, decimals=6, show_fractions=True, col_width=12)
