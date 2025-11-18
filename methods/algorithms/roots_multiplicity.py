# methods/algorithms/roots_multiplicity.py
from sympy import symbols, sympify, lambdify
import numpy as np

def _fmt_x(x):
    return f"{x:.10f}" if abs(x) >= 1e-6 else f"{x:.4e}"

def _fmt_sci(y):
    return f"{y:.1e}"

def roots_multiplicity(x0, f_str, f1_str, f2_str, tol=1e-7, max_iter=100):
    """
    Multiple roots (modified Newton):
      x_{k+1} = x_k - (f*f') / ((f')^2 - f*f'')
    Prints the iteration table and returns the approximation.
    """
    x = symbols('x')
    allowed = {"x": x}

    f   = lambdify(x, sympify(f_str,  locals=allowed), modules=("numpy",))
    fp  = lambdify(x, sympify(f1_str, locals=allowed), modules=("numpy",))
    fpp = lambdify(x, sympify(f2_str, locals=allowed), modules=("numpy",))

    print("Multiple Roots\n")
    print("Results Table:\n")
    print("| iter|     xi         |   f(xi)   |     E     |")

    f0 = float(f(x0))
    print(f"|  0  | {_fmt_x(x0):<14} | {_fmt_sci(f0):>7}  |           |")

    for k in range(1, max_iter + 1):
        fx   = float(f(x0))
        fpx  = float(fp(x0))
        fppx = float(fpp(x0))

        num = fx * fpx
        den = fpx*fpx - fx*fppx

        if abs(num) < 1e-30:
            x1 = x0
            E  = 0.0
            fx1 = float(f(x1))
            print(f"|  {k:<1}  | {_fmt_x(x1):<14} | {_fmt_sci(fx1):>7}  | {_fmt_sci(E):>9}  |")
            print(f"\nAn approximation of the root was found at {x1:.12f}")
            print("_________________________________________________________________")
            return x1

        if abs(den) < 1e-15:
            print("\nDenominator ~ 0; cannot proceed safely.")
            return x0

        x1 = x0 - (num/den)
        E  = abs(x1 - x0)
        fx1 = float(f(x1))

        print(f"|  {k:<1}  | {_fmt_x(x1):<14} | {_fmt_sci(fx1):>7}  | {_fmt_sci(E):>9}  |")

        if E < tol:
            print(f"|  {k+1:<1}  | {_fmt_x(x1):<14} | {_fmt_sci(float(f(x1))):>7}  | {_fmt_sci(0.0):>9}  |")
            print(f"\nAn approximation of the root was found at {x1:.12f}")
            print("_________________________________________________________________")
            return x1

        x0 = x1

    print("\nThe maximum number of iterations was reached without meeting tolerance.")
    return x0

# Already compatible with invoke_root_algorithm
