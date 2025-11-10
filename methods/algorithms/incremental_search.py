# incremental_search.py
import math
from sympy import symbols, sympify, lambdify

# ---------- Safe function builder ----------
def build_function(func_str: str):
    """
    Converts a string like 'sin(x) - x/2' into a numeric callable f(x).
    Uses SymPy for safe parsing and lambdify for fast evaluation.
    """
    x = symbols('x')
    expr = sympify(func_str)
    f = lambdify(x, expr, modules=['math'])
    return f, str(expr)

# ---------- Incremental Search (collects ALL brackets) ----------
def incremental_search(x0: float, delta: float, niter: int, f, tol: float = 1e-12):
    """
    Incremental Search method to find all intervals [x, x+delta] where f(x)*f(x+delta) < 0.
    It scans niter steps starting from x0 and accumulates all brackets and exact roots.
    """
    steps = []
    intervals = []
    exact_roots = []

    x = float(x0)
    fx = float(f(x))

    if abs(fx) <= tol:
        exact_roots.append(x)

    for k in range(1, niter + 1):
        x_next = x + delta
        fx_next = float(f(x_next))

        steps.append({"k": k, "x": x, "fx": fx, "x_next": x_next, "fx_next": fx_next})

        if abs(fx_next) <= tol:
            exact_roots.append(x_next)

        if fx * fx_next < 0:
            a, b = (x, x_next) if x < x_next else (x_next, x)
            intervals.append((a, b))

        x, fx = x_next, fx_next

    return {
        "status": "done",
        "intervals": intervals,
        "exact_roots": exact_roots,
        "iterations": len(steps),
        "steps": steps,
    }

# ---------- Pretty table printer ----------
def print_steps_table(steps, floatfmt=".10g"):
    headers = ["Iter", "x", "f(x)", "x_next", "f(x_next)"]

    def fnum(v):
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

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    def hline(char="-", cross="+"):
        return cross + cross.join(char * (w + 2) for w in widths) + cross

    def fmt_row(values):
        cells = [str(v).ljust(w) for v, w in zip(values, widths)]
        return "| " + " | ".join(cells) + " |"

    print(hline("-", "+"))
    print(fmt_row(headers))
    print(hline("=", "+"))
    for r in rows:
        print(fmt_row(r))
    print(hline("-", "+"))

# ---------- Main program ----------
def main():
    print("=== Incremental Search Method ===\n")

    # Predefined test parameters (from the image)
    func_str = "log(sin(x)**2 + 1) - 1/2"   # f(x) = ln(sin²(x)+1) - 1/2
    x0 = -3                                # initial x
    delta = 0.5                            # step size
    niter = 100                            # max iterations
    tol = 1e-7                             # tolerance

    f, normalized = build_function(func_str)
    print(f"Using function: f(x) = {normalized}")
    print(f"x0 = {x0}, Δx = {delta}, N = {niter}, Tol = {tol}\n")

    # Run method
    result = incremental_search(x0, delta, niter, f, tol)

    print("Iteration Table:")
    print_steps_table(result["steps"], floatfmt=".10g")

    # Results
    print("\nIncremental Search Results\n")
    if result["intervals"]:
        for (a, b) in result["intervals"]:
            print(f"Root found in interval [{a:.10f}, {b:.10f}]")
    else:
        print("No sign change intervals found.")

    for r in result["exact_roots"]:
        print(f"Exact root detected at x = {r:.10f}")

if __name__ == "__main__":
    main()
