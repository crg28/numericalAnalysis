# methods/views/linear_utils.py
from __future__ import annotations

import io
import importlib
import inspect
from contextlib import redirect_stdout

from sympy import Matrix


# -----------------------------
# Matrix / vector parsing
# -----------------------------
def parse_matrix_flex(text: str) -> Matrix:
    """
    Parse a text block into a Sympy Matrix.
    Accepts formats like:
        1 2 3
        4 5 6

    or:
        1, 2, 3; 4 5 6

    or:
        [[1, 2, 3], [4, 5, 6]]
    """
    t = (text or "").strip()
    if not t:
        return Matrix([])

    # Python literal list
    if t.startswith("["):
        from ast import literal_eval
        return Matrix(literal_eval(t))

    rows = []
    for raw in t.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = [p for p in raw.replace(",", " ").split() if p]
        rows.append([float(x) for x in parts])

    # Single line with semicolons: "1 2 3; 4 5 6"
    if len(rows) == 1 and ";" in t:
        rows = []
        for seg in t.split(";"):
            parts = [p for p in seg.replace(",", " ").split() if p]
            if parts:
                rows.append([float(x) for x in parts])

    return Matrix(rows)


def parse_vector_flex(text: str) -> Matrix:
    """
    Parse a text block into a column vector (Sympy Matrix with shape (n, 1)).

    Accepts formats like:
        1
        2
        3

    or:
        1, 2, 3
        1 2 3
        [1, 2, 3]
    """
    t = (text or "").strip()
    if not t:
        return Matrix([])

    # Python literal list
    if t.startswith("["):
        from ast import literal_eval
        data = literal_eval(t)
        return Matrix(data).reshape(len(data), 1)

    vals = []
    if "\n" in t:
        # One value per line
        for line in t.splitlines():
            line = line.strip()
            if line:
                vals.append(float(line.replace(",", " ").split()[0]))
    else:
        # Single line with spaces/semicolons/commas
        vals = [
            float(p)
            for p in t.replace("\n", " ").replace(";", " ").replace(",", " ").split()
            if p
        ]

    return Matrix(vals).reshape(len(vals), 1)


def fmt_matrix(M):
    """
    Convert a Sympy Matrix (or list-like) into a nested Python float list.
    Useful to convert to numpy arrays afterwards.
    """
    if M is None:
        return None
    try:
        return [[float(v) for v in row] for row in list(M.tolist())]
    except Exception:
        return [[float(v) for v in row] for row in M]


# -----------------------------
# Mapping: method.kind -> module file
# -----------------------------
MODULE_NAME_OVERRIDES = {
    # kind in DB / Method.kind   ->   real python module (without .py)
    "pivot_partial": "partial_pivoting",
    "pivot_total": "total_pivoting",
    "lu_simple": "simple_lu_factorization",
    "lu_pivot": "pivoting_lu_factorization",
    # SOR: kind = "sor" but file is SOR.py
    "sor": "SOR",
}


# -----------------------------
# Candidate function names
# -----------------------------
_LINEAR_FUN_CANDIDATES = (
    # Iterative methods
    "jacobi", "jacobi_matricial",
    "gauss_seidel",
    "sor", "sor_run", "SOR",

    # LU / factorization
    "crout_demo", "crout",
    "doolittle_demo", "doolittle",

    # Cholesky
    "cholesky_demo", "cholesky_like", "cholesky",

    # Elimination + LU
    "gaussian_elimination", "simple_gaussian_elimination",
    "gaussian_elimination_partial_pivoting", "partial_pivoting",
    "gaussian_elimination_total_pivoting", "total_pivoting",
    "simple_lu_factorization",
    "pivoting_lu_factorization",
    "lu", "lu_factorization",

    # Genéricos al final (para no molestar)
    "run", "solve", "algorithm", "main", "execute",
)



def _pick_linear_callable(mod):
    for name in _LINEAR_FUN_CANDIDATES:
        if hasattr(mod, name):
            return getattr(mod, name), name
    return None, None


def invoke_linear_algorithm(kind: str, A: Matrix, b: Matrix,
                            extras: dict | None = None) -> str | None:
    """
    Import methods/algorithms/<kind>.py (or mapped name) and capture EXACT stdout.

    Automatically adapts to functions that:
      - only need A  (e.g. cholesky_like(A))
      - need A and b (e.g. gaussian_elimination(A, b))
      - need A, b, tol, max_iter, w, ... (e.g. jacobi, gauss_seidel, sor)
    """
    # Map kind -> real module name if needed
    module_name = MODULE_NAME_OVERRIDES.get(kind, kind)
    module_name = module_name.replace("-", "_")

    try:
        mod = importlib.import_module(f"methods.algorithms.{module_name}")
    except Exception as e:
        raise ImportError(f"Could not import module 'methods.algorithms.{module_name}': {e}")

    fn, fn_name = _pick_linear_callable(mod)
    if not fn:
        raise AttributeError(
            f"No suitable function found in module '{module_name}'. "
            f"Tried: {', '.join(_LINEAR_FUN_CANDIDATES)}"
        )

    import numpy as np

    A_np = np.array(fmt_matrix(A), dtype=float)
    b_np = np.array([float(v) for v in list(b)], dtype=float)

    extras = extras or {}

    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())
    has_b = "b" in param_names  # does this function expect b?

    def extras_for_signature():
        filtered = {}
        for k, v in extras.items():
            if k in param_names:
                filtered[k] = v
        return filtered

    call_variants = []

    if has_b:
        # --- Functions that expect A and b (e.g. gaussian_elimination, cholesky_demo, sor) ---
        call_variants.append(
            ({"_pos": [A_np, b_np]}, extras_for_signature())
        )
        call_variants.append(
            ({"_pos": [A_np, b_np]}, {})
        )
        if "A" in param_names and "b" in param_names:
            call_variants.append(
                ({}, {"A": A_np, "b": b_np, **extras_for_signature()})
            )
            call_variants.append(
                ({}, {"A": A_np, "b": b_np})
            )
        if "extras" in param_names:
            call_variants.append(
                ({"_pos": [A_np, b_np]}, {"extras": extras})
            )
    else:
        # --- Functions that only depend on A (e.g. cholesky_like(A)) ---
        call_variants.append(
            ({"_pos": [A_np]}, extras_for_signature())
        )
        call_variants.append(
            ({"_pos": [A_np]}, {})
        )
        if "A" in param_names:
            call_variants.append(
                ({}, {"A": A_np, **extras_for_signature()})
            )
            call_variants.append(
                ({}, {"A": A_np})
            )
        if "extras" in param_names:
            call_variants.append(
                ({"_pos": [A_np]}, {"extras": extras})
            )

    buf = io.StringIO()
    last_error = None

    for args_dict, kwargs in call_variants:
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                if "_pos" in args_dict:
                    fn(*args_dict["_pos"], **kwargs)
                else:
                    fn(**kwargs)
            break
        except Exception as e:
            last_error = e
            continue

    out = buf.getvalue()

    if not out and last_error is not None:
        # propagate last error so it can be shown in ctx["error"]
        raise last_error

    return out if out else None
