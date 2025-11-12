# methods/utils.py
from __future__ import annotations
import io
import importlib
from contextlib import redirect_stdout
from sympy import Matrix, sympify, lambdify, symbols

# ================================================================
# Symbolic compiler for single-var functions f(x)
# ================================================================
x = symbols("x")

def compile_fx(expr_text: str):
    """Compile text into a numeric single-variable function f(x)."""
    expr = sympify(expr_text, convert_xor=True)
    f = lambdify(x, expr, "numpy")
    return f, expr

# ================================================================
# Flexible parsing for matrices and vectors
# ================================================================
def parse_matrix_flex(text: str) -> Matrix:
    """
    Accept Excel-like rows, CSV, or Python-like [[...],[...]] to build a Sympy Matrix.
    """
    t = (text or "").strip()
    if not t:
        return Matrix([])
    if t.startswith("["):
        from ast import literal_eval
        data = literal_eval(t)
        return Matrix(data)

    rows = []
    for raw in t.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = [p for p in raw.replace(",", " ").split() if p]
        if len(parts) == 1 and "," in raw:
            parts = [p for p in raw.split(",") if p.strip()]
        rows.append([float(x) for x in parts])

    if len(rows) == 1 and ";" in t:
        rows = []
        for seg in t.split(";"):
            parts = [p for p in seg.replace(",", " ").split() if p]
            if parts:
                rows.append([float(x) for x in parts])

    return Matrix(rows)

def parse_vector_flex(text: str) -> Matrix:
    """
    Accepts [1,2,3], newline-separated values, or CSV; returns a column vector.
    """
    t = (text or "").strip()
    if not t:
        return Matrix([])
    if t.startswith("["):
        from ast import literal_eval
        data = literal_eval(t)
        return Matrix(data).reshape(len(data), 1)

    vals = []
    if "\n" in t:
        for line in t.splitlines():
            line = line.strip()
            if not line:
                continue
            vals.append(float(line.replace(",", " ").split()[0]))
    else:
        vals = [float(p.strip()) for p in t.replace("\n", " ").replace(";", " ").split(",") if p.strip()]
        if not vals:
            vals = [float(p) for p in t.split() if p]

    return Matrix(vals).reshape(len(vals), 1)

def fmt_matrix(M):
    """Convert Matrix or ndarray to list[list[float]] for templates."""
    if M is None:
        return None
    try:
        return [[float(v) for v in row] for row in list(M.tolist())]
    except Exception:
        return [[float(v) for v in row] for row in M]

# ================================================================
# Session helpers (last inputs)
# ================================================================
def load_last(request, kind: str) -> dict:
    bucket = request.session.get("last_inputs", {})
    return bucket.get(kind, {})

def save_last(request, kind: str, cleaned: dict):
    bucket = request.session.get("last_inputs", {})
    bucket[kind] = {k: cleaned.get(k) for k in cleaned}
    request.session["last_inputs"] = bucket

# ================================================================
# External algorithm invoker (captures stdout)
# ================================================================
_CANDIDATE_FUNCS = (
    "run", "solve", "algorithm", "main", "execute",
    "cholesky_demo", "cholesky_like", "cholesky",
    "crout", "doolittle", "jacobi", "gauss_seidel", "sor",
    "gaussian_elimination", "partial_pivoting", "total_pivoting",
    "lu", "lu_factorization",
)

def invoke_algorithm(kind: str, A: Matrix, b: Matrix, extras: dict | None = None) -> str | None:
    """
    Import methods.algorithms.{kind} and run its main function,
    capturing everything printed to stdout.
    """
    try:
        module_name = kind.replace("-", "_")
        mod = importlib.import_module(f"methods.algorithms.{module_name}")
    except Exception:
        return None

    func = None
    for fname in _CANDIDATE_FUNCS:
        if hasattr(mod, fname):
            func = getattr(mod, fname)
            break
    if func is None:
        return None

    import numpy as np
    A_np = np.array(fmt_matrix(A), dtype=float)
    b_np = np.array([float(v) for v in list(b)], dtype=float)

    kwargs = {}
    if extras:
        kwargs["extras"] = extras

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            func(A_np, b_np, **kwargs)
        return buf.getvalue()
    except Exception:
        return buf.getvalue() or None
