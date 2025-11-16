import math
import sympy as sp

x = sp.symbols("x")


def newton(f, x0=0.0, tol=1e-7, max_iter=100, df_str=None, **kwargs):
    """
    Método de Newton-Raphson adaptado a invoke_root_algorithm.

    Criterio de paro:
      - Se detiene cuando el error E = |x_i - x_{i-1}| < tol
        (o cuando se alcanzan max_iter iteraciones).
    """

    # 1. Verificar que tenemos f y df_str
    if f is None:
        print("Newton: no se recibió la función f.")
        return

    if not df_str:
        print("Newton: se requiere la derivada f'(x) (df_str).")
        return

    # 2. Construir df callable a partir de df_str
    try:
        df_expr = sp.sympify(df_str)
    except Exception as e:
        print(f"Newton: no se pudo interpretar df_str='{df_str}': {e}")
        return

    df = sp.lambdify(x, df_expr, "math")

    # 3. Cabecera de la tabla
    print("Método de Newton–Raphson\n")
    print("Tabla de resultados:\n")
    print("| iter|          xi |       f(xi) |          E |")

    xi = float(x0)
    x_prev = None

    for i in range(max_iter):
        fxi = f(xi)

        if i == 0:
            # primera iteración → no hay E aún
            print(f"| {i:3d} | {xi:11.9f} | {fxi:11.1e} |            |")
        else:
            E = abs(xi - x_prev)
            print(f"| {i:3d} | {xi:11.9f} | {fxi:11.1e} | {E:10.1e} |")

            # 🔸 Criterio de paro por E
            if E < tol:
                break

        # Paso de Newton
        try:
            dfxi = df(xi)
        except Exception as e:
            print(f"\nError al evaluar f'({xi}): {e}")
            return

        if dfxi == 0:
            print(f"\nDerivada nula en x = {xi:.9f}. No se puede continuar.")
            return

        x_prev = xi
        xi = xi - fxi / dfxi

    print(f"\nSe encontró una aproximación de la raíz en {xi}")
