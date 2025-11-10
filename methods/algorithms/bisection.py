# bisection_method.py
import numpy as np
from sympy import symbols, sympify, lambdify, sin, cos, tan, log, exp
import re

# ---------- Utility to fix function powers ----------
def rewrite_function_powers(s: str) -> str:
    """
    Fix patterns like:
        sin**2(x) -> (sin(x))**2
        cos**3(2*x)-> (cos(2*x))**3
    """
    i = 0
    out = ""
    while i < len(s):
        if s[i].isalpha():
            j = i + 1
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            func_name = s[i:j]
            if s[j:j + 2] == "**":
                k = j + 2
                m = k
                while m < len(s) and s[m].isdigit():
                    m += 1
                if m > k and m < len(s) and s[m] == "(":
                    power_str = s[k:m]
                    start_arg = m
                    depth = 0
                    pos = start_arg
                    end_arg = None
                    while pos < len(s):
                        if s[pos] == "(":
                            depth += 1
                        elif s[pos] == ")":
                            depth -= 1
                            if depth == 0:
                                end_arg = pos
                                break
                        pos += 1
                    if end_arg is None:
                        out += s[i]
                        i += 1
                        continue
                    arg_str = s[start_arg:end_arg + 1]
                    out += f"({func_name}{arg_str})**{power_str}"
                    i = end_arg + 1
                    continue
        out += s[i]
        i += 1
    return out

# ---------- Normalize math syntax ----------
def normalize_function_string(f_str_raw: str) -> str:
    f = f_str_raw.strip()
    f = f.replace("^", "**")
    f = re.sub(r"\bln\b", "log", f)
    f = rewrite_function_powers(f)
    return f

# ---------- Build numeric function ----------
def build_numeric_function(f_str_raw: str):
    x = symbols("x")
    f_clean = normalize_function_string(f_str_raw)
    allowed_locals = {"x": x, "sin": sin, "cos": cos, "tan": tan, "log": log, "exp": exp, "np": np}
    expr = sympify(f_clean, locals=allowed_locals)
    f_num = lambdify(x, expr, modules=["numpy"])
    return f_num, expr, f_clean

# ---------- Bisection Method ----------
def bisection(a, b, f_str, tol, max_iter):
    f_num, expr, cleaned = build_numeric_function(f_str)

    fa = f_num(a)
    fb = f_num(b)

    if np.isnan(fa) or np.isnan(fb):
        print("Error: f(a) or f(b) is NaN (domain issue).")
        return None
    if fa * fb > 0:
        print("Error: f(a) and f(b) must have opposite signs for bisection.")
        print(f"f({a}) = {fa}, f({b}) = {fb}")
        return None

    print("\nBisection Method\n")
    print(f"Function        : f(x) = {cleaned}")
    print(f"Sympy form      : {expr}")
    print(f"Interval        : a = {a}, b = {b}")
    print(f"Tolerance       : {tol}")
    print(f"Max Iterations  : {max_iter}\n")

    print("| Iter |        a        |        c        |        b        |        f(c)        |     Error      |")
    print("-" * 100)

    c_old = (a + b) / 2.0
    fc_old = f_num(c_old)
    print(f"| {1:4d} | {a:12.6f} | {c_old:12.6f} | {b:12.6f} | {fc_old:16.9e} | {'-':>13s} |")

    for i in range(2, max_iter + 1):
        fa = f_num(a)
        fc = f_num(c_old)
        if fa * fc < 0:
            b = c_old
        else:
            a = c_old
        c_new = (a + b) / 2.0
        fc_new = f_num(c_new)
        err = abs(c_new - c_old)
        print(f"| {i:4d} | {a:12.6f} | {c_new:12.6f} | {b:12.6f} | {fc_new:16.9e} | {err:13.6e} |")
        if err < tol:
            print("\nApproximate root found:")
            print(f"  x ≈ {c_new:.10f}")
            print(f"  f(x) = {fc_new:.3e}")
            print(f"  after {i} iterations (tolerance reached).\n")
            return c_new
        c_old = c_new

    print("\nIteration limit reached without meeting tolerance.")
    print(f"Last approximation: x ≈ {c_old:.10f}, f(x) = {f_num(c_old):.3e}\n")
    return c_old

# ---------- Main Test Case ----------
if __name__ == "__main__":
    # Test values from the problem
    f_str = "ln(sin^2(x) + 1) - 1/2"
    a = 0
    b = 1
    tol = 1e-7
    N = 100

    bisection(a, b, f_str, tol, N)
