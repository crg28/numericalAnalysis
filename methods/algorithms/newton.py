import math
import sympy as sp

x = sp.symbols("x")


def newton(f, x0=0.0, tol=1e-7, max_iter=100, df_str=None, **kwargs):
    """
    Newton–Raphson Method adapted for invoke_root_algorithm.

    Stopping criterion:
      - The method stops when the error E = |x_i - x_{i-1}| < tol
        (or when max_iter iterations are reached).
    """

    # 1. Verify we received f and df_str
    if f is None:
        print("Newton: function f was not provided.")
        return

    if not df_str:
        print("Newton: the derivative f'(x) (df_str) is required.")
        return

    # 2. Build df callable from df_str
    try:
        df_expr = sp.sympify(df_str)
    except Exception as e:
        print(f"Newton: could not interpret df_str='{df_str}': {e}")
        return

    df = sp.lambdify(x, df_expr, "math")

    # 3. Table header
    print("Newton–Raphson Method\n")
    print("Results Table:\n")
    print("| iter|          xi |       f(xi) |          E |")

    xi = float(x0)
    x_prev = None

    for i in range(max_iter):
        fxi = f(xi)

        if i == 0:
            # first iteration → no error E yet
            print(f"| {i:3d} | {xi:11.9f} | {fxi:11.1e} |            |")
        else:
            E = abs(xi - x_prev)
            print(f"| {i:3d} | {xi:11.9f} | {fxi:11.1e} | {E:10.1e} |")

            # 🔸 Stop criterion
            if E < tol:
                break

        # Newton step
        try:
            dfxi = df(xi)
        except Exception as e:
            print(f"\nError evaluating f'({xi}): {e}")
            return

        if dfxi == 0:
            print(f"\nZero derivative at x = {xi:.9f}. Cannot continue.")
            return

        x_prev = xi
        xi = xi - fxi / dfxi

    print(f"\nAn approximation of the root was found at {xi}")
