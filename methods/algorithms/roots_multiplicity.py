from sympy import symbols, sympify, lambdify
import numpy as np

def _fmt_x(x):
    return f"{x:.10f}" if abs(x) >= 1e-6 else f"{x:.4e}"

def _fmt_sci(y):
    return f"{y:.1e}"

def rootsMultiplicity(x0, f_str, f1_str, f2_str, tol=1e-7, max_ite=100):
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

    print("Raíces Múltiples\n")
    print("Tabla de resultados:\n")
    print("| iter|     xi         |   f(xi)   |     E     |")

    # Row 0
    f0 = float(f(x0))
    print(f"|  0  | {_fmt_x(x0):<14} | {_fmt_sci(f0):>7}  |           |")

    for k in range(1, max_ite + 1):
        fx   = float(f(x0))
        fpx  = float(fp(x0))
        fppx = float(fpp(x0))

        # Compute numerator and denominator
        num = fx * fpx
        den = fpx*fpx - fx*fppx

        # If numerator ~ 0, we're essentially at a fixed point -> x_{k+1} = x_k
        if abs(num) < 1e-30:
            x1 = x0
            E  = 0.0
            fx1 = float(f(x1))
            print(f"|  {k:<1}  | {_fmt_x(x1):<14} | {_fmt_sci(fx1):>7}  | {_fmt_sci(E):>9}  |")
            print(f"\nSe encontró una aproximación de la raiz en {x1:.12f}")
            print("_________________________________________________________________")
            return x1

        # If denominator ~ 0 and numerator is not ~0, then it's unsafe to continue
        if abs(den) < 1e-15:
            print("\nDenominador ~ 0; no se puede continuar con seguridad.")
            return x0

        # Normal modified-Newton step
        x1 = x0 - (num/den)
        E  = abs(x1 - x0)
        fx1 = float(f(x1))

        # Print iteration row with the NEW x
        print(f"|  {k:<1}  | {_fmt_x(x1):<14} | {_fmt_sci(fx1):>7}  | {_fmt_sci(E):>9}  |")

        if E < tol:
            # Extra row with E = 0 as in your white screenshot
            print(f"|  {k+1:<1}  | {_fmt_x(x1):<14} | {_fmt_sci(float(f(x1))):>7}  | {_fmt_sci(0.0):>9}  |")
            print(f"\nSe encontró una aproximación de la raiz en {x1:.12f}")
            print("_________________________________________________________________")
            return x1

        x0 = x1

    print("\nSe alcanzó el límite de iteraciones sin cumplir la tolerancia.")
    return x0

# Use the required functions and initial guess:
rootsMultiplicity(
    1.0,
    "exp(x) - x - 1",
    "exp(x) - 1",
    "exp(x)",
    tol=1e-7,
    max_ite=100
)
