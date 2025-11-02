import numpy as np
from sympy import symbols, sympify, lambdify, sin, cos, tan, log, exp
import re

def rewrite_function_powers(s: str) -> str:
    """
    Fix patterns like:
        sin**2(x)  -> (sin(x))**2
        cos**3(2*x)-> (cos(2*x))**3
    after we've already replaced ^ -> **.
    We'll scan for <name>**<int>(<arg>) and rewrite.
    """

    i = 0
    out = ""

    while i < len(s):
        # try to match: <funcname>**<power>(
        if s[i].isalpha():
            # read function name
            j = i + 1
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            func_name = s[i:j]

            # need "**"
            if s[j:j+2] == "**":
                k = j + 2
                # read integer exponent
                m = k
                while m < len(s) and s[m].isdigit():
                    m += 1
                # must have at least one digit
                if m > k and m < len(s) and s[m] == "(":
                    power_str = s[k:m]     # e.g. "2"
                    # now capture the full (...) argument
                    start_arg = m          # points at '('
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
                        # malformed parentheses, just fall back to default
                        out += s[i]
                        i += 1
                        continue

                    arg_str = s[start_arg:end_arg+1]  # "( ... )"

                    # build "(func_name(arg))**power"
                    out += f"({func_name}{arg_str})**{power_str}"

                    # advance i to after the closing ')'
                    i = end_arg + 1
                    continue
                # else: pattern didn't fully match, fall through

        # default: just copy char
        out += s[i]
        i += 1

    return out


def normalize_function_string(f_str_raw: str) -> str:
    """
    Convert user-friendly math into valid SymPy syntax.

    Steps:
    - replace '^' with '**'
    - replace 'ln' with 'log'
    - rewrite things like sin**2(x) into (sin(x))**2
    """
    f = f_str_raw.strip()

    # 1) ^ -> **
    f = f.replace("^", "**")

    # 2) ln -> log  (natural log)
    #    NOTE: do NOT replace inside variable names, so use regex
    f = re.sub(r"\bln\b", "log", f)

    # 3) fix function powers like sin**2(x)
    f = rewrite_function_powers(f)

    return f


def build_numeric_function(f_str_raw: str):
    """
    Take the raw user input string, sanitize it, parse with SymPy,
    and return a numeric callable f(x).
    Also returns the SymPy expr and the cleaned string.
    """
    x = symbols("x")

    f_clean = normalize_function_string(f_str_raw)

    allowed_locals = {
        "x": x,
        "sin": sin,
        "cos": cos,
        "tan": tan,
        "log": log,  # natural log
        "exp": exp,
        "np": np,
    }

    expr = sympify(f_clean, locals=allowed_locals)
    f_num = lambdify(x, expr, modules=["numpy"])
    return f_num, expr, f_clean


def bisection(a, b, f_str, tol, max_iter):
    """
    Bisection method to approximate a root of f(x) in [a,b].

    Stopping rule:
        error = |c_new - c_old| < tol
    or we hit max_iter.

    Prints an iteration table and returns the last approximation.
    """

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
    print(f"Original input : {f_str}")
    print(f"Interpreted as : {cleaned}")
    print(f"SymPy form     : {expr}")
    print(f"Interval       : a = {a}, b = {b}")
    print(f"Tolerance      : {tol}")
    print(f"Max iterations : {max_iter}\n")

    print("| Iter |        a        |        c        |        b        |        f(c)        |     Error      |")
    print("-" * 100)

    # iteration 1 (initial midpoint)
    c_old = (a + b) / 2.0
    fc_old = f_num(c_old)
    print(f"| {1:4d} | {a:12.6f} | {c_old:12.6f} | {b:12.6f} | {fc_old:16.9e} | {'-':>13s} |")

    # loop
    for i in range(2, max_iter + 1):
        fa = f_num(a)
        fc = f_num(c_old)

        # choose subinterval
        if fa * fc < 0:
            b = c_old
        else:
            a = c_old

        # new midpoint
        c_new = (a + b) / 2.0
        fc_new = f_num(c_new)

        # absolute error between midpoints
        err = abs(c_new - c_old)

        print(f"| {i:4d} | {a:12.6f} | {c_new:12.6f} | {b:12.6f} | {fc_new:16.9e} | {err:13.6e} |")

        if err < tol:
            print("\nApproximate root found:")
            print(f"  x ≈ {c_new:.10f}")
            print(f"  f(x) = {fc_new:.3e}")
            print(f"  after {i} iterations (tolerance reached).\n")
            return c_new

        c_old = c_new

    # not converged
    print("\nIteration limit reached without meeting tolerance.")
    print(f"Last approximation: x ≈ {c_old:.10f}, f(x) = {f_num(c_old):.3e}\n")
    return c_old


def read_user_input():
    """
    Interactive prompt for:
    - f(x)
    - a, b
    - tolerance
    - max iterations
    """
    print("=== Bisection Method ===\n")
    print("Enter the function f(x). Examples:")
    print("  x^3 - x - 1")
    print("  ln(sin^2(x)+1) - 1/2")
    print("  exp(x) - 4*x")
    print("  sin(x) - 0.5")
    print()

    f_str = input("f(x) = ")

    a = float(input("Enter a (left endpoint): "))
    b = float(input("Enter b (right endpoint): "))

    tol = float(input("Enter tolerance (example 1e-6): "))
    max_iter = int(input("Enter maximum number of iterations N: "))

    return a, b, f_str, tol, max_iter


if __name__ == "__main__":
    a, b, f_str, tol, max_iter = read_user_input()
    bisection(a, b, f_str, tol, max_iter)
