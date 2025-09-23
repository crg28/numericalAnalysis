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
            raise np.linalg.LinAlgError("Pivote ~0 durante back_substitution (total).")
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
    all_lines = []
    for heading, lines in sections:
        all_lines.append("▸ " + heading)
        all_lines.extend(lines if lines else ["(vacío)"])
        all_lines.append("")
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

# ===================== Pivoteo Total =====================

def solve_total_pivot(A, b, tol=1e-12):
    """
    Factoriza con pivoteo total (filas y columnas):  P A Q = L U
    Devuelve x, U, P, Q.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("A debe ser cuadrada (total).")
    if b.shape[0] != n:
        raise ValueError("Dimensión de b incompatible (total).")

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)   # permutación de filas
    Q = np.eye(n)   # permutación de columnas

    for k in range(n - 1):
        # Mayor |.| en submatriz U[k:,k:]
        sub = np.abs(U[k:, k:])
        i_rel, j_rel = np.unravel_index(int(np.argmax(sub)), sub.shape)
        i = k + i_rel
        j = k + j_rel

        if abs(U[i, j]) < tol:
            raise np.linalg.LinAlgError(f"Matriz singular o pivote ~0 en submatriz k={k} (total).")

        # swap filas
        if i != k:
            U[[k, i], :] = U[[i, k], :]
            P[[k, i], :] = P[[i, k], :]
            if k > 0:
                L[[k, i], :k] = L[[i, k], :k]

        # swap columnas
        if j != k:
            U[:, [k, j]] = U[:, [j, k]]
            Q[:, [k, j]] = Q[:, [j, k]]

        # Eliminación
        piv = U[k, k]
        for i2 in range(k + 1, n):
            m = U[i2, k] / piv
            L[i2, k] = m
            U[i2, k:] -= m * U[k, k:]

    # Resolver: P A Q z = L U z = P b  =>  y=L^{-1}(Pb),  U z = y,  x = Q z
    Pb = P @ b
    y  = forward_substitution(L, Pb)
    z  = back_substitution(U, y, tol=tol)
    x  = Q @ z
    return x, U, P, Q

def print_solution(title, A0, b, x, U, P, Q, decimals=6, show_fractions=True, col_width=12):
    residuo = float(np.linalg.norm(A0 @ x - b, 2))
    condA   = float(np.linalg.cond(A0, 2))

    # Solución
    sol_lines = []
    for i, val in enumerate(x, start=1):
        base = f"x{i} = {float(val):.{decimals}f}"
        if show_fractions:
            base += f"   ({Fraction(float(val)).limit_denominator()})"
        sol_lines.append(base)

    # Métricas
    metrics_lines = [
        f"‖A x − b‖₂ = {residuo:.3e}",
        f"cond₂(A)   = {condA:.3e}",
        f"dtype      = {A0.dtype}",
    ]

    # PAQ y U
    PAQ = P @ A0 @ Q
    paq_lines = _matrix_lines(PAQ, decimals=decimals, col_width=col_width)
    u_lines   = _matrix_lines(U,   decimals=decimals, col_width=col_width)

    sections = [
        ("Solución", sol_lines),
        ("Métricas", metrics_lines),
        ("Matriz tras pivoteo (PAQ)", paq_lines),
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
    x, U, P, Q = solve_total_pivot(A, b)
    print_solution("Pivoteo Total", A0, b, x, U, P, Q, decimals=6, show_fractions=True, col_width=12)
