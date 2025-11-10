from sympy import symbols, sympify, lambdify, sin, log
import numpy as np

def false_position(a, b, f_str, tol=1e-7, max_iter=100, stop_on_residual=False, residual_tol=1e-12):
    """
    False Position (Regula Falsi).
    Prints: iter | a | xm | b | f(xm) | E
    Stops by default ONLY when step error E = |x_n - x_{n-1}| < tol.
    If stop_on_residual=True, it will also stop when |f(xm)| < residual_tol.
    """
    x = symbols('x')
    allowed = {"x": x, "sin": sin, "log": log, "ln": log}
    expr = sympify(f_str, locals=allowed)
    f = lambdify(x, expr, modules="numpy")

    fa, fb = float(f(a)), float(f(b))
    if np.sign(fa) * np.sign(fb) >= 0:
        raise ValueError("The interval [a, b] does not bracket a root (f(a)*f(b) >= 0).")

    print("False position\n")
    print("Results table:\n")
    print("| iter|          a |          xm |           b |     f(xm) |           E |")

    c_prev = None
    for k in range(1, max_iter + 1):
        fa, fb = float(f(a)), float(f(b))
        c = b - fb * (b - a) / (fb - fa)
        fc = float(f(c))

        # step error
        E = "" if c_prev is None else f"{abs(c - c_prev):.1e}"
        print(f"| {k:>4d}| {a: .10f} | {c: .10f} | {b: .10f} | {fc: .1e} | {E:>11} |")

        # stop criteria
        if c_prev is not None and abs(c - c_prev) < tol:
            print(f"\nAn approximation of the root was found at {c:.12f}")
            return c
        if stop_on_residual and abs(fc) < residual_tol:
            print(f"\nAn approximation of the root was found at {c:.12f}")
            return c

        # update bracket
        if fa * fc < 0:
            b = c
        else:
            a = c

        c_prev = c

    print("\nMaximum number of iterations reached.")
    return None


# Test case (matches your sheet: should print 5 rows)
if __name__ == "__main__":
    false_position(0, 1, "ln(sin(x)**2 + 1) - 1/2", tol=1e-7, max_iter=100)
