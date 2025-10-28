from django.contrib.auth import login
from django.contrib.auth.views import LoginView, LogoutView
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

from sympy import sympify, lambdify, symbols, Matrix, diff

from .forms import SignUpForm, NonlinearForm, LinearSystemForm
from .models import Category, Method


# ---------------------- Utilidades comunes ----------------------
def _common_ctx():
    return {"categories": Category.objects.all().order_by("name")}


# ---------------------- Home / Listado / Búsqueda ----------------------
def home(request):
    ctx = _common_ctx()
    ctx["featured_methods"] = (
        Method.objects.filter(is_featured=True)[:8]
        if hasattr(Method, "is_featured")
        else Method.objects.all()[:8]
    )
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
    return render(request, "methods/list.html", ctx)  # 👈 reutilizamos list.html


# Mantengo el nombre para que coincida con base.html
def search_products(request):
    return method_list(request)


# ---------------------- Auth ----------------------
def signup(request):
    if request.user.is_authenticated:
        return redirect("methods:home")
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # no es staff
            return redirect("methods:home")
    else:
        form = SignUpForm()
    return render(request, "registration/signup.html", {"form": form})


class AnalysisLoginView(LoginView):
    template_name = "registration/login.html"


class AnalysisLogoutView(LogoutView):
    pass


# ---------------------- Runners numéricos ----------------------
x = symbols("x")


def _compile_fx(expr_text):
    expr = sympify(expr_text, convert_xor=True)
    f = lambdify(x, expr, "numpy")
    return f, expr


def _bisection(f, a, b, tol, maxit):
    rows = []
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None, [{"error": "f(a)*f(b) >= 0"}]
    for k in range(1, maxit + 1):
        c = (a + b) / 2
        fc = f(c)
        rows.append({"k": k, "a": a, "b": b, "c": c, "f(c)": fc, "err": abs(b - a) / 2})
        if abs(fc) < tol or abs(b - a) / 2 < tol:
            return c, rows
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return c, rows


def _newton(expr, x0, tol, maxit):
    f = lambdify(x, expr, "numpy")
    df = lambdify(x, diff(expr, x), "numpy")
    rows = []
    xk = x0
    for k in range(1, maxit + 1):
        fx, dfx = f(xk), df(xk)
        if dfx == 0:
            rows.append({"k": k, "x": xk, "f(x)": fx, "df(x)": dfx, "error": "df=0"})
            break
        xnew = xk - fx / dfx
        err = abs(xnew - xk)
        rows.append({"k": k, "x": xk, "f(x)": fx, "df(x)": dfx, "x_next": xnew, "err": err})
        xk = xnew
        if err < tol or abs(fx) < tol:
            break
    return xk, rows


def _secant(f, x0, x1, tol, maxit):
    rows = []
    a, b = x0, x1
    fa, fb = f(a), f(b)
    for k in range(1, maxit + 1):
        if fb - fa == 0:
            rows.append({"k": k, "x0": a, "x1": b, "error": "div/0"})
            break
        x2 = b - fb * (b - a) / (fb - fa)
        err = abs(x2 - b)
        rows.append({"k": k, "x0": a, "x1": b, "x2": x2, "f(x2)": f(x2), "err": err})
        a, fa, b, fb = b, fb, x2, f(x2)
        if err < tol or abs(fb) < tol:
            break
    return b, rows


def _fixed_point(gexpr, x0, tol, maxit):
    g = lambdify(x, gexpr, "numpy")
    rows = []
    xk = x0
    for k in range(1, maxit + 1):
        xnew = g(xk)
        err = abs(xnew - xk)
        rows.append({"k": k, "x": xk, "x_next": xnew, "err": err})
        xk = xnew
        if err < tol:
            break
    return xk, rows


def method_run(request, slug):
    method = get_object_or_404(Method, slug=slug)

    # No lineales
    if method.kind in ("bisection", "newton", "secant", "fixed_point"):
        form = NonlinearForm(request.POST or None)
        context = {"method": method, "form": form, "iters": None, "root": None, "error": None}
        if request.method == "POST" and form.is_valid():
            data = form.cleaned_data
            fx_txt = data.get("function") or data.get("funcion")
            tol = float(data["tol"])
            maxit = int(data["max_iter"])
            try:
                if method.kind == "fixed_point":
                    g_txt = data.get("g_function") or data.get("gfuncion")
                    gexpr = sympify(g_txt)
                    root, rows = _fixed_point(gexpr, float(data["x0"]), tol, maxit)
                else:
                    f, expr = _compile_fx(fx_txt)
                    if method.kind == "bisection":
                        root, rows = _bisection(f, float(data["a"]), float(data["b"]), tol, maxit)
                    elif method.kind == "newton":
                        root, rows = _newton(expr, float(data["x0"]), tol, maxit)
                    elif method.kind == "secant":
                        root, rows = _secant(f, float(data["x0"]), float(data["x1"]), tol, maxit)
                context.update({"root": root, "iters": rows})
            except Exception as e:
                context["error"] = str(e)
        return render(request, "methods/run_nonlinear.html", context)

    # Lineales (pivot)
    if method.kind in ("pivot_partial", "pivot_total"):
        form = LinearSystemForm(request.POST or None)
        context = {"method": method, "form": form, "solution": None, "steps": None}
        if request.method == "POST" and form.is_valid():
            A_txt = form.cleaned_data["A"]
            b_txt = form.cleaned_data["b"]
            A = Matrix([[float(x) for x in row.split(",")] for row in A_txt.split(";")])
            bvec = Matrix([float(x) for x in b_txt.split(",")])
            sol = A.LUsolve(bvec)  # pivoting implícito en LU
            context["solution"] = [float(s) for s in sol]
        return render(request, "methods/run_linear.html", context)

    # Fallback si algún tipo distinto existiera
    return render(request, "methods/detail.html", {"method": method})
