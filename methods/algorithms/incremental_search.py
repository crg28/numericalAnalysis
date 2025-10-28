# incremental_search.py
import math
from sympy import symbols, sympify, lambdify

# ---------- Safe parsing of the function ----------
def build_function(func_str: str):
    """
    Turn a string like 'sin(x) - x/2' into a numeric function f(x).
    Uses SymPy to parse safely and lambdify to get a fast callable.
    """
    x = symbols('x')
    try:
        expr = sympify(func_str)  # safe parse (no eval)
    except Exception as e:
        raise ValueError(f"Invalid function: {e}")
    # use standard math module (radians)
    f = lambdify(x, expr, modules=['math'])
    return f, str(expr)

# ---------- Incremental Search algorithm ----------
def incremental_search(x0: float, delta: float, niter: int, f, tol: float = 1e-12):
    """
    Incremental Search for roots of f(x) = 0.
    - x0: starting point
    - delta: step size (can be negative to search left)
    - niter: max iterations
    - f: callable f(x)
    - tol: tolerance to consider a value as a root

    Returns a dict with status: 'root' | 'bracket' | 'no-root' and a list of steps.
    """
    if delta == 0:
        raise ValueError("Delta must be non-zero.")
    if niter <= 0:
        raise ValueError("Max iterations must be positive.")

    steps = []
    x = float(x0)
    fx = float(f(x))

    # Check if the starting point is already (approximately) a root
    if abs(fx) <= tol:
        return {"status": "root", "root": x, "iterations": 0, "steps": [{"k": 0, "x": x, "fx": fx}]}

    for k in range(1, niter + 1):
        x_next = x + delta
        fx_next = float(f(x_next))

        # Keep a record of this iteration
        steps.append({"k": k, "x": x, "fx": fx, "x_next": x_next, "fx_next": fx_next})

        # Exact root (within tolerance) at the new point
        if abs(fx_next) <= tol:
            return {"status": "root", "root": x_next, "iterations": k, "steps": steps}

        # Sign change ⇒ a root is bracketed between x and x_next
        if fx * fx_next < 0:
            a, b = (x, x_next) if x < x_next else (x_next, x)
            return {"status": "bracket", "a": a, "b": b, "iterations": k, "steps": steps}

        # Advance
        x, fx = x_next, fx_next

    # No sign change detected within the iteration budget
    return {"status": "no-root", "iterations": niter, "last_x": x, "last_fx": fx, "steps": steps}

# ---------- Pretty table printer (no external deps) ----------
def print_steps_table(steps, floatfmt=".10g"):
    """
    Pretty-print iteration steps as a table with | separators.
    steps: list of dicts with keys k, x, fx, x_next, fx_next (some may be missing)
    """
    headers = ["Iteration", "x", "f(x)", "x_next", "f(x_next)"]

    def fnum(v):
        # format floats consistently, accept non-floats too
        try:
            return f"{float(v):{floatfmt}}"
        except Exception:
            return str(v)

    rows = []
    for s in steps:
        rows.append([
            s.get("k", ""),
            fnum(s.get("x", "")),
            fnum(s.get("fx", "")),
            fnum(s.get("x_next", "")),
            fnum(s.get("fx_next", "")),
        ])

    # column widths = max(header, cells)
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    def hline(char="-", cross="+"):
        return cross + cross.join(char * (w + 2) for w in widths) + cross

    def fmt_row(values):
        cells = [str(v).ljust(w) for v, w in zip(values, widths)]
        return "| " + " | ".join(cells) + " |"

    # print table
    print(hline("-", "+"))
    print(fmt_row(headers))
    print(hline("=", "+"))
    for r in rows:
        print(fmt_row(r))
    print(hline("-", "+"))

# ---------- Simple CLI helpers ----------
def ask_float(prompt):
    """Ask for a float until the user provides a valid number."""
    while True:
        s = input(prompt).strip()
        try:
            return float(s)
        except ValueError:
            print("Please enter a valid number.")

def ask_int(prompt):
    """Ask for a positive integer until the user provides one."""
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if v > 0:
                return v
            print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")

def main():
    print("=== Incremental Search Method for f(x) = 0 ===")
    print("Enter functions like: sin(x) - x/2, x**3 - 7*x + 1, exp(x) - 3")
    func_str = input("f(x) = ").strip()

    # Build a safe callable from user input
    try:
        f, normalized = build_function(func_str)
        print(f"Parsed function: f(x) = {normalized}")
    except ValueError as e:
        print(e)
        return

    # Collect parameters
    x0 = ask_float("x0            = ")
    delta = ask_float("Δx (step)    = ")
    niter = ask_int("max iterations = ")

    # Run the algorithm
    try:
        result = incremental_search(x0, delta, niter, f)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Pretty table output
    print()
    print_steps_table(result["steps"], floatfmt=".10g")

    # Final message
    print("\nResult:")
    status = result["status"]
    if status == "root":
        print(f"  Found a root at x = {result['root']:.12g} (iterations: {result['iterations']})")
    elif status == "bracket":
        print(f"  A root is bracketed in ({result['a']:.12g}, {result['b']:.12g}) "
              f"(iterations: {result['iterations']})")
    else:
        print(f"  No sign change found within {result['iterations']} iterations. "
              f"Last x = {result['last_x']:.12g}, f(x) = {result['last_fx']:.12g}")

if __name__ == "__main__":
    main()
