# methods/views/views_roots.py
from __future__ import annotations
import io, base64
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from django import forms
from django.shortcuts import render, get_object_or_404

from sympy import symbols, sympify, lambdify
from numpy import linspace  # optional

from ..models import Method
from .utils import compile_fx  # use shared helper


x = symbols("x")

# --------- Forms for each root-finding method ---------
class BisectionForm(forms.Form):
    fx = forms.CharField(label="f(x)", initial="x**3 + 4*x**2 - 10")
    a  = forms.FloatField(label="a")
    b  = forms.FloatField(label="b")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)

class NewtonForm(forms.Form):
    fx = forms.CharField(label="f(x)")
    x0 = forms.FloatField(label="x0")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)

class SecantForm(forms.Form):
    fx = forms.CharField(label="f(x)")
    x0 = forms.FloatField(label="x0")
    x1 = forms.FloatField(label="x1")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)

class FixedPointForm(forms.Form):
    gx = forms.CharField(label="g(x)")
    x0 = forms.FloatField(label="x0")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)

class FalsePositionForm(forms.Form):
    fx = forms.CharField(label="f(x)")
    a  = forms.FloatField(label="a")
    b  = forms.FloatField(label="b")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)

class IncrementalSearchForm(forms.Form):
    fx = forms.CharField(label="f(x)")
    x0 = forms.FloatField(label="x0")
    delta = forms.FloatField(label="Δ", initial=0.5)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)

class MultipleRootsForm(forms.Form):
    fx = forms.CharField(label="f(x)")
    x0 = forms.FloatField(label="x0")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)


# --------- local helpers ---------
def _compile_fx_local(expr_text):
    expr = sympify(expr_text, convert_xor=True)
    return lambdify(x, expr, "numpy"), expr


def _bisection(f, a, b, tol, maxit):
    rows = []
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None, [{"error": "f(a) * f(b) >= 0 (no sign change)."}]
    c = a
    for k in range(1, maxit + 1):
        c_old = c
        c = (a + b) / 2
        fc = f(c)
        rows.append({"k": k, "a": a, "b": b, "c": c, "f(c)": fc, "err": abs(c - c_old)})
        if abs(fc) < tol or abs(c - c_old) < tol:
            break
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return c, rows


def _newton(expr, x0, tol, maxit):
    f = lambdify(x, expr, "numpy")
    df = lambdify(x, expr.diff(x), "numpy")
    rows = []
    xk = x0
    for k in range(1, maxit + 1):
        fx, dfx = f(xk), df(xk)
        if dfx == 0:
            rows.append({"k": k, "x": xk, "f": fx, "error": "f'(xk) = 0"})
            break
        xnew = xk - fx / dfx
        rows.append({"k": k, "x": xk, "x_next": xnew, "err": abs(xnew - xk)})
        if abs(fx) < tol or abs(xnew - xk) < tol:
            xk = xnew
            break
        xk = xnew
    return xk, rows


# --------- help texts ---------
HELP = {
    "bisection": [
        "Requires f(a)·f(b) < 0.",
        "Plot includes a, b, and the final c marker.",
    ],
    "newton": [
        "Requires f and f'. Stops if f'(xk)=0.",
    ],
    "secant": ["Does not require derivative."],
    "fixed_point": ["Convergence if |g'(x)| < 1 in a neighborhood."],
    "false_position": ["Bracketing using linear interpolation."],
    "incremental_search": ["Scans for sign changes over steps."],
    "multiple_roots": ["Typically uses f, f', and f'' (if implemented)."],
}


def _form_for_kind(kind, *args, **kwargs):
    if kind == "bisection": return BisectionForm(*args, **kwargs)
    if kind == "newton": return NewtonForm(*args, **kwargs)
    if kind == "secant": return SecantForm(*args, **kwargs)
    if kind == "fixed_point": return FixedPointForm(*args, **kwargs)
    if kind == "false_position": return FalsePositionForm(*args, **kwargs)
    if kind == "incremental_search": return IncrementalSearchForm(*args, **kwargs)
    if kind == "multiple_roots": return MultipleRootsForm(*args, **kwargs)
    return BisectionForm(*args, **kwargs)


def method_run_roots(request, slug):
    method = get_object_or_404(Method, slug=slug)
    kind = method.kind

    form = _form_for_kind(kind, request.POST or None)

    ctx = {
        "method": method,
        "form": form,
        "help_items": HELP.get(kind, []),
        "table": None,
        "root": None,
        "plot_data_uri": None,
        "error": None,
    }

    if request.method == "POST" and form.is_valid():
        try:
            if kind == "bisection":
                f, _ = _compile_fx_local(form.cleaned_data["fx"])
                a = float(form.cleaned_data["a"])
                b = float(form.cleaned_data["b"])
                tol = float(form.cleaned_data["tol"])
                n  = int(form.cleaned_data["max_iter"])
                root, rows = _bisection(f, a, b, tol, n)
                ctx["root"] = root
                ctx["table"] = rows

                # Plot f(x) on [a,b] with markers at a, b, and c*
                try:
                    import numpy as np
                    X = np.linspace(a, b, 400)
                    Y = f(X)
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.axhline(0, linewidth=0.8)
                    ax.plot(X, Y)
                    if root is not None:
                        ax.scatter([a, b, root], [f(a), f(b), f(root)], s=30)
                    ax.set_title("Bisection")
                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png", dpi=120)
                    plt.close(fig)
                    ctx["plot_data_uri"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
                except Exception:
                    pass

            elif kind == "newton":
                _, expr = _compile_fx_local(form.cleaned_data["fx"])
                x0 = float(form.cleaned_data["x0"])
                tol = float(form.cleaned_data["tol"])
                n  = int(form.cleaned_data["max_iter"])
                root, rows = _newton(expr, x0, tol, n)
                ctx["root"] = root
                ctx["table"] = rows

            else:
                ctx["error"] = "UI for this root-finding method is not implemented yet. You can hook your own algorithm module as done for linear methods."

        except Exception as e:
            ctx["error"] = str(e)

    return render(request, "methods/run_roots.html", ctx)
