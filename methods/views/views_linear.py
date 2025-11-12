# methods/views/views_linear.py
from __future__ import annotations
import importlib, io
from contextlib import redirect_stdout
from django.shortcuts import render, get_object_or_404
from sympy import Matrix
from ..models import Method
from ..forms import AbForm, IterativeForm, SorForm
from .utils import parse_matrix_flex, parse_vector_flex, fmt_matrix, invoke_algorithm



# ---------- parsing helpers ----------
def _parse_matrix_flex(text: str) -> Matrix:
    t = (text or "").strip()
    if not t:
        return Matrix([])
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
    if len(rows) == 1 and ";" in t:
        rows = []
        for seg in t.split(";"):
            parts = [p for p in seg.replace(",", " ").split() if p]
            if parts:
                rows.append([float(x) for x in parts])
    return Matrix(rows)


def _parse_vector_flex(text: str) -> Matrix:
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
            if line:
                vals.append(float(line.replace(",", " ").split()[0]))
    else:
        vals = [float(p) for p in t.replace("\n", " ").replace(";", " ").replace(",", " ").split() if p]
    return Matrix(vals).reshape(len(vals), 1)


def _fmt_matrix(M):
    if M is None:
        return None
    try:
        return [[float(v) for v in row] for row in list(M.tolist())]
    except Exception:
        return [[float(v) for v in row] for row in M]


# ---------- stdout capture for your algorithms modules ----------
_CANDIDATE_FUNCS = (
    "run", "solve", "algorithm", "main", "execute",
    "cholesky_demo", "cholesky_like", "cholesky",
    "crout", "doolittle", "jacobi", "gauss_seidel", "sor",
    "gaussian_elimination", "partial_pivoting", "total_pivoting",
    "lu", "lu_factorization",
)

def _invoke_algorithm(kind: str, A: Matrix, b: Matrix, extras=None) -> str | None:
    try:
        mod = importlib.import_module(f"methods.algorithms.{kind.replace('-', '_')}")
    except Exception:
        return None
    fn = None
    for name in _CANDIDATE_FUNCS:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            break
    if not fn:
        return None

    import numpy as np
    A_np = np.array(_fmt_matrix(A), dtype=float)
    b_np = np.array([float(v) for v in list(b)], dtype=float)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            if extras:
                fn(A_np, b_np, extras=extras)
            else:
                fn(A_np, b_np)
        return buf.getvalue()
    except Exception:
        return buf.getvalue() or None


# ---------- help texts ----------
HELP_TEXT = {
    "jacobi": [
        "Converges if A is diagonally dominant or SPD.",
        "Shown error is the residual norm ||Ax-b|| per iteration.",
    ],
    "gauss_seidel": [
        "In-place updates; often faster than Jacobi.",
    ],
    "sor": [
        "0 < ω < 2; ω = 1 ⇒ Gauss-Seidel.",
    ],
    "crout": ["LU with diag(U)=1 (Crout)."],
    "doolittle": ["LU with diag(L)=1 (Doolittle)."],
    "cholesky": [
        "Requires SPD (A=Aᵀ, xᵀAx>0). Falls back to LU if not SPD.",
    ],
    "gaussian_elimination": ["Gaussian Elimination."],
    "pivot_partial": ["Partial pivoting by rows."],
    "pivot_total": ["Total pivoting by rows and columns."],
    "lu_simple": ["Plain LU (no pivoting)."],
    "lu_pivot": ["LU with pivoting."],
}


def method_run_linear(request, slug):
    method = get_object_or_404(Method, slug=slug)
    kind = method.kind

    # Select form by method kind
    if kind in ("jacobi", "gauss_seidel"):
        form = IterativeForm(request.POST or None)
    elif kind == "sor":
        form = SorForm(request.POST or None)
    else:
        form = AbForm(request.POST or None)

    ctx = {
        "method": method,
        "form": form,
        "help_items": HELP_TEXT.get(kind, []),
        "solution": None, "L": None, "U": None, "P": None,
        "iters": None, "steps": None, "console": None, "error": None,
    }

    if request.method == "POST" and form.is_valid():
        try:
            A = _parse_matrix_flex(form.cleaned_data.get("A", ""))
            b = _parse_vector_flex(form.cleaned_data.get("b", ""))

            # Extras passed into your algorithms
            extras = {}
            if kind in ("jacobi", "gauss_seidel", "sor"):
                extras["tol"] = float(form.cleaned_data.get("tol", 1e-6))
                extras["max_iter"] = int(form.cleaned_data.get("max_iter", 50))
                x0_txt = form.cleaned_data.get("x0") or ""
                x0 = _parse_vector_flex(x0_txt)
                extras["x0"] = [float(v) for v in list(x0)] if x0.shape[0] else [0.0] * A.shape[0]
            if kind == "sor":
                extras["w"] = float(form.cleaned_data.get("w", 1.0))

            console = _invoke_algorithm(kind, A, b, extras)
            if console:
                ctx["console"] = console

            # Base solution and optional L/U/P
            try:
                ctx["solution"] = [float(v) for v in list(A.LUsolve(b))]
            except Exception:
                ctx["solution"] = None

            if kind in ("doolittle", "crout", "lu_simple", "lu_pivot",
                        "gaussian_elimination", "pivot_partial", "pivot_total", "cholesky"):
                try:
                    if kind == "cholesky":
                        try:
                            L = A.cholesky(); U = L.T; P = None
                        except Exception:
                            L, U, perm = A.LUdecomposition()
                            from sympy import Matrix as M
                            P = M.eye(A.shape[0])
                            for i, j in enumerate(perm):
                                P.row_swap(i, j)
                    else:
                        L, U, perm = A.LUdecomposition()
                        from sympy import Matrix as M
                        P = M.eye(A.shape[0])
                        for i, j in enumerate(perm):
                            P.row_swap(i, j)

                    ctx["L"] = _fmt_matrix(L)
                    ctx["U"] = _fmt_matrix(U)
                    ctx["P"] = _fmt_matrix(P) if P is not None else None
                except Exception:
                    pass

        except Exception as e:
            ctx["error"] = str(e)

    return render(request, "methods/run_linear.html", ctx)
