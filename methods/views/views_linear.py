# methods/views/views_linear.py
from __future__ import annotations

from django.shortcuts import render, get_object_or_404
from sympy import Matrix

from ..models import Method
from ..forms import AbForm, IterativeForm, SorForm
from .linear_utils import (
    parse_matrix_flex,
    parse_vector_flex,
    fmt_matrix,
    invoke_linear_algorithm,
)


# ---------- help text ----------
HELP_TEXT = {
    "jacobi": [
        "Convergence if A is diagonally dominant or SPD.",
        "Shows the residual ||Ax-b|| per iteration.",
    ],
    "gauss_seidel": [
        "Updates in-place; often converges faster than Jacobi.",
    ],
    "sor": [
        "0<ω<2; ω=1 ⇒ Gauss-Seidel.",
    ],
    "crout": ["LU with diag(U)=1 (Crout)."],
    "doolittle": ["LU with diag(L)=1 (Doolittle)."],
    "cholesky": [
        "Requires SPD (A=Aᵀ, xᵀAx>0). If not SPD, falls back to LU.",
    ],
    "gaussian_elimination": ["Gaussian Elimination."],
    "pivot_partial": ["Partial pivoting by rows."],
    "pivot_total": ["Total pivoting by rows and columns."],
    "lu_simple": ["LU factorization without pivoting."],
    "lu_pivot": ["LU factorization with pivoting."],
}


def method_run_linear(request, slug):
    """
    Single entry for all linear-system/factorization methods.
    It invokes your existing algorithms in methods/algorithms/*
    and captures EXACTLY their console output.
    """
    method = get_object_or_404(Method, slug=slug)
    kind = method.kind  # e.g., 'jacobi', 'crout', 'cholesky', ...

    # Pick the right form
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
        "solution": None,
        "L": None,
        "U": None,
        "P": None,
        "console": None,
        "error": None,
    }

    if request.method == "POST" and form.is_valid():
        try:
            A = parse_matrix_flex(form.cleaned_data.get("A", ""))
            b = parse_vector_flex(form.cleaned_data.get("b", ""))

            # Build extras for iterative methods
            extras = {}
            if kind in ("jacobi", "gauss_seidel", "sor"):
                extras["tol"] = float(form.cleaned_data.get("tol", 1e-6))
                extras["max_iter"] = int(form.cleaned_data.get("max_iter", 50))

                x0_vec = parse_vector_flex(form.cleaned_data.get("x0") or "")
                extras["x0"] = (
                    [float(v) for v in list(x0_vec)]
                    if x0_vec.shape[0]
                    else [0.0] * A.shape[0]
                )

            if kind == "sor":
                extras["w"] = float(form.cleaned_data.get("w", 1.0))

            # ---- invoke your module and capture EXACT console output
            console = invoke_linear_algorithm(kind, A, b, extras)
            if console:
                ctx["console"] = console

            # Optional: provide a direct SymPy solve as a convenience
            try:
                ctx["solution"] = [float(v) for v in list(A.LUsolve(b))]
            except Exception:
                ctx["solution"] = None

            # If the method naturally yields L/U/P, try to expose them with SymPy
            if kind in (
                "doolittle",
                "crout",
                "lu_simple",
                "lu_pivot",
                "gaussian_elimination",
                "pivot_partial",
                "pivot_total",
                "cholesky",
            ):
                try:
                    if kind == "cholesky":
                        try:
                            L = A.cholesky()
                            U = L.T
                            P = None
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

                    ctx["L"] = fmt_matrix(L)
                    ctx["U"] = fmt_matrix(U)
                    ctx["P"] = fmt_matrix(P) if P is not None else None
                except Exception:
                    pass

        except Exception as e:
            ctx["error"] = str(e)

    return render(request, "methods/run_linear.html", ctx)
