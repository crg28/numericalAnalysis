from __future__ import annotations
import importlib
import io
from contextlib import redirect_stdout

from django.contrib.auth import login
from django.contrib.auth.views import LoginView, LogoutView
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

from sympy import Matrix, sympify, lambdify, symbols
from .forms import SignUpForm, AbForm, IterativeForm, SorForm
from .models import Category, Method

# ================================================================
# Utilidades de contexto
# ================================================================
def _common_ctx():
    return {"categories": Category.objects.all().order_by("name")}

# ================================================================
# Home / listado / búsqueda
# ================================================================
def home(request):
    ctx = _common_ctx()
    ctx["featured_methods"] = Method.objects.all().order_by("name")[:8]
    return render(request, "methods/home.html", ctx)

def method_list(request):
    ctx = _common_ctx()
    q = request.GET.get("q", "").strip()
    qs = Method.objects.all()
    if q:
        qs = qs.filter(Q(name__icontains=q) | Q(description__icontains=q))
    ctx["methods"] = qs.order_by("name")
    return render(request, "methods/list.html", ctx)

def category_view(request, slug):
    ctx = _common_ctx()
    cat = get_object_or_404(Category, slug=slug)
    ctx["category"] = cat
    ctx["methods"] = Method.objects.filter(category=cat).order_by("name")
    return render(request, "methods/list.html", ctx)

def search_products(request):
    return method_list(request)

# ================================================================
# Auth
# ================================================================
def signup(request):
    if request.user.is_authenticated:
        return redirect("methods:home")
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("methods:home")
    else:
        form = SignUpForm()
    return render(request, "registration/signup.html", {"form": form})

class AnalysisLoginView(LoginView):
    template_name = "registration/login.html"

class AnalysisLogoutView(LogoutView):
    pass

# ================================================================
# Helpers de parsing y sesión (historial)
# ================================================================
def _parse_matrix_flex(text: str) -> Matrix:
    """
    Acepta:
      - Formato tipo Excel: filas separadas por salto de línea y columnas por espacios
      - CSV '1,2,3; 4,5,6'
      - Python-like: [[1,2],[3,4]]
    """
    t = (text or "").strip()
    if not t:
        return Matrix([])
    # Intento literal (JSON/Python)
    if t.startswith("["):
        from ast import literal_eval
        data = literal_eval(t)
        return Matrix(data)
    # Normalizamos: reemplazar múltiples espacios por uno,
    # permite tanto “1 2 3” como “1, 2, 3”
    rows = []
    for raw in t.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if ";" in raw and "," not in raw:
            # raro: línea con ; — tratamos globalmente abajo
            pass
        # primero intentamos por espacios
        parts = [p for p in raw.replace(",", " ").split() if p]
        if len(parts) == 1 and "," in raw:
            parts = [p for p in raw.split(",") if p.strip()]
        rows.append([float(x) for x in parts])
    # Si solo hay 1 fila pero contiene ';', es “fila1; fila2; …”
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
    # soporta líneas “1\n2\n3”, o “1,2,3”
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

def _fmt_matrix(M) -> list[list[float]]:
    """Convierte Sympy Matrix o numpy array a lista de listas para la plantilla."""
    if M is None:
        return None
    try:
        return [[float(v) for v in row] for row in list(M.tolist())]
    except Exception:
        # fallback para np.ndarray
        return [[float(v) for v in row] for row in M]

def _load_last(request, kind: str) -> dict:
    bucket = request.session.get("last_inputs", {})
    return bucket.get(kind, {})

def _save_last(request, kind: str, cleaned: dict):
    bucket = request.session.get("last_inputs", {})
    # guardamos solo campos presentes
    bucket[kind] = {k: cleaned.get(k) for k in cleaned}
    request.session["last_inputs"] = bucket

# ================================================================
# Invocador de algoritmos externos (captura stdout)
# ================================================================
_CANDIDATE_FUNCS = (
    "run", "solve", "algorithm", "main", "execute",
    # algunos nombres usados en tu repo:
    "cholesky_demo", "cholesky_like", "cholesky",
    "crout", "doolittle", "jacobi", "gauss_seidel", "sor",
    "gaussian_elimination", "partial_pivoting", "total_pivoting",
    "lu", "lu_factorization",
)

def _invoke_algorithm(kind: str, A: Matrix, b: Matrix, extras: dict | None = None) -> str | None:
    """
    Importa methods.algorithms.{kind} (con '-' -> '_') y ejecuta la primera
    función disponible de _CANDIDATE_FUNCS capturando lo que imprime a stdout.
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

    # Prepara argumentos simples
    import numpy as np
    A_np = np.array(_fmt_matrix(A), dtype=float)
    b_np = np.array([float(v) for v in list(b)], dtype=float)

    # extras llega desde el form (tol, max_iter, x0, w, norm, etc.)
    kwargs = {}
    if extras:
        kwargs["extras"] = extras

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            # La mayoría acepta: func(A, b, extras=...)
            func(A_np, b_np, **kwargs)
        return buf.getvalue()
    except Exception:
        return buf.getvalue() or None

# ================================================================
# Textos de ayuda por método
# ================================================================
HELP_TEXT = {
    "jacobi": [
        "Convergencia usual si A es estrictamente diagonal dominante o simétrica definida positiva.",
        "Error mostrado = norma del residuo ||Ax-b||.",
    ],
    "gauss_seidel": [
        "Similar a Jacobi pero usa actualizaciones in-place; suele converger más rápido.",
        "Convergencia bajo A SDD o SPD.",
    ],
    "sor": [
        "Requiere ω. 0 < ω < 2. ω>1 = sobre-relajación.",
        "Si ω=1 se reduce a Gauss-Seidel.",
    ],
    "doolittle": ["Factoriza A ≈ L·U con diag(L)=1 (Doolittle)."],
    "crout": ["Factoriza A ≈ L·U con diag(U)=1 (Crout)."],
    "cholesky": [
        "Requiere matriz simétrica definida positiva (A=Aᵀ, xᵀAx>0).",
        "Si A no es SPD, se mostrará advertencia y se dará solución por LU.",
    ],
    "gaussian_elimination": ["Eliminación Gaussiana con/ sin pivoteo según variante."],
    "pivot_partial": ["Pivoteo parcial por filas."],
    "pivot_total": ["Pivoteo total por filas y columnas."],
    "lu_simple": ["LU simple (sin pivoteo)."],
    "lu_pivot": ["LU con pivoteo parcial."],
}

# ================================================================
# Ejecutar método (vista principal de cálculo)
# ================================================================
def method_run(request, slug):
    method = get_object_or_404(Method, slug=slug)
    kind = method.kind

    # Selección de formulario + historial
    initial = _load_last(request, kind)
    if kind in ("jacobi", "gauss_seidel"):
        form = IterativeForm(request.POST or None, initial=initial)
    elif kind == "sor":
        form = SorForm(request.POST or None, initial=initial)
    else:
        form = AbForm(request.POST or None, initial=initial)

    context = {
        "method": method,
        "form": form,
        "help_items": HELP_TEXT.get(kind, []),
        "solution": None,
        "L": None,
        "U": None,
        "P": None,
        "iters": None,
        "steps": None,
        "console": None,
        "error": None,
    }

    if request.method == "POST" and form.is_valid():
        _save_last(request, kind, form.cleaned_data)
        try:
            A = _parse_matrix_flex(form.cleaned_data.get("A", ""))
            b = _parse_vector_flex(form.cleaned_data.get("b", ""))

            # Extras comunes para invocadores
            extras = {}
            if kind in ("jacobi", "gauss_seidel", "sor"):
                extras["tol"] = float(form.cleaned_data.get("tol", 1e-6))
                extras["max_iter"] = int(form.cleaned_data.get("max_iter", 50))
                x0_txt = form.cleaned_data.get("x0") or ""
                x0 = _parse_vector_flex(x0_txt)
                extras["x0"] = [float(v) for v in list(x0)] if x0.shape[0] else [0.0] * A.shape[0]
            if kind == "sor":
                extras["w"] = float(form.cleaned_data.get("w", 1.0))
            if hasattr(form.cleaned_data, "get") and form.cleaned_data.get("norm"):
                extras["norm"] = form.cleaned_data.get("norm")

            # 1) Intentamos llamar un algoritmo propio y capturar su consola
            console = _invoke_algorithm(kind, A, b, extras)
            if console:
                context["console"] = console

            # 2) Siempre damos una solución numérica base (LUsolve) para mostrar x
            try:
                x_sol = [float(v) for v in list(A.LUsolve(b))]
                context["solution"] = x_sol
            except Exception:
                context["solution"] = None

            # 3) Si es factorization-kind, mostramos L/U/P usando Sympy
            if kind in ("doolittle", "crout", "lu_simple", "lu_pivot",
                        "gaussian_elimination", "pivot_partial", "pivot_total", "cholesky"):
                try:
                    if kind == "cholesky":
                        try:
                            L = A.cholesky()
                            U = L.T
                            P = None
                        except Exception as _:
                            # No SPD: usamos LU genérica (con permutación)
                            L, U, perm = A.LUdecomposition()
                            P = Matrix.eye(A.shape[0])
                            for i, j in enumerate(perm):
                                P.row_swap(i, j)
                            context["help_items"] = [
                                "A no es simétrica definida positiva. Se mostró LU como alternativa."
                            ] + context["help_items"]
                    else:
                        L, U, perm = A.LUdecomposition()
                        P = Matrix.eye(A.shape[0])
                        for i, j in enumerate(perm):
                            P.row_swap(i, j)

                    context["L"] = _fmt_matrix(L)
                    context["U"] = _fmt_matrix(U)
                    context["P"] = _fmt_matrix(P) if P is not None else None
                except Exception:
                    # no pasa nada, al menos mostramos la solución
                    pass

        except Exception as e:
            context["error"] = str(e)

    return render(request, "methods/run_linear.html", context)

# ================================================================
# (Opcional) helpers de funciones de una variable si llegas a usarlas
# ================================================================
x = symbols("x")
def _compile_fx(expr_text):
    expr = sympify(expr_text, convert_xor=True)
    f = lambdify(x, expr, "numpy")
    return f, expr
