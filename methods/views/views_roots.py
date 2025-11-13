# methods/views/views_roots.py
from __future__ import annotations

import io
import base64

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from django import forms
from django.shortcuts import render, get_object_or_404

from sympy import symbols

from ..models import Method
from .utils import compile_fx, invoke_root_algorithm

x = symbols("x")


# ------------------------- Forms -------------------------
class BisectionForm(forms.Form):
    fx = forms.CharField(label="f(x)", initial="x**3 + 4*x**2 - 10")
    a = forms.FloatField(label="a")
    b = forms.FloatField(label="b")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)


class FalsePositionForm(forms.Form):
    fx = forms.CharField(label="f(x)", initial="x**3 + 4*x**2 - 10")
    a = forms.FloatField(label="a")
    b = forms.FloatField(label="b")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)


class FixedPointForm(forms.Form):
    gx = forms.CharField(label="g(x)")
    x0 = forms.FloatField(label="x0")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)


class IncrementalSearchForm(forms.Form):
    fx = forms.CharField(label="f(x)")
    x0 = forms.FloatField(label="x0")
    delta = forms.FloatField(label="Δ", initial=0.5)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)


class MultipleRootsForm(forms.Form):
    fx = forms.CharField(label="f(x)")
    dfx = forms.CharField(label="f'(x)")
    d2fx = forms.CharField(label="f''(x)")
    x0 = forms.FloatField(label="x0")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)


class NewtonForm(forms.Form):
    fx = forms.CharField(label="f(x)")
    dfx = forms.CharField(label="f'(x)")
    x0 = forms.FloatField(label="x0")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)


class SecantForm(forms.Form):
    fx = forms.CharField(label="f(x)")
    x0 = forms.FloatField(label="x0")
    x1 = forms.FloatField(label="x1")
    tol = forms.FloatField(label="Tolerance", initial=1e-6)
    max_iter = forms.IntegerField(label="Max iterations", initial=50)


# ------------------------- Form selector -------------------------
def _form_for_kind(kind, *args, **kwargs):
    if kind == "bisection":
        return BisectionForm(*args, **kwargs)
    if kind == "false_position":
        return FalsePositionForm(*args, **kwargs)
    if kind == "fixed_point":
        return FixedPointForm(*args, **kwargs)
    if kind == "incremental_search":
        return IncrementalSearchForm(*args, **kwargs)
    if kind == "multiple_roots":
        return MultipleRootsForm(*args, **kwargs)
    if kind == "newton":
        return NewtonForm(*args, **kwargs)
    if kind == "secant":
        return SecantForm(*args, **kwargs)
    # default
    return BisectionForm(*args, **kwargs)


# ------------------------- Plot helpers -------------------------
def _plot_fx_interval(f, a: float, b: float, title: str) -> str | None:
    """
    Plot f(x) on [a,b] with horizontal axis and return a data URI (png).
    """
    try:
        X = np.linspace(a, b, 400)
        Y = f(X)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axhline(0, linewidth=0.8)
        ax.plot(X, Y)
        ax.set_title(title)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# ------------------------- Main view -------------------------
HELP = {
    "bisection": [
        "Requires f(a)·f(b) < 0.",
        "Plot included on [a,b].",
    ],
    "false_position": [
        "Bracketing with linear interpolation.",
        "Plot included on [a,b].",
    ],
    "fixed_point": ["Converges if |g'(x)| < 1 nearby."],
    "incremental_search": ["Search for sign change by steps of Δ."],
    "multiple_roots": ["Uses f, f' and f''."],
    "newton": ["Requires f and f'."],
    "secant": ["No derivative required."],
}

def method_run_roots(request, slug):
    """
    Unified roots runner:
    - Builds expr_text (string), f_lambda (callable) and numeric params
    - Calls invoke_root_algorithm(kind, expr_text, f_lambda, params)
    - Captures exact console output from your algorithms module
    - Draws plot only for bracket methods [a,b]
    """
    method = get_object_or_404(Method, slug=slug)
    kind = (method.kind or "").lower()  # e.g., "bisection", "false_position", ...

    form = _form_for_kind(kind, request.POST or None)

    ctx = {
        "method": method,
        "form": form,
        "help_items": HELP.get(kind, []),
        "console": None,          # exact console output from your module
        "error": None,            # exception message if any
        "plot_data_uri": None,    # plot (for [a,b] methods)
    }

    if request.method == "POST" and form.is_valid():
        try:
            cleaned = form.cleaned_data
            expr_text: str = ""     # textual function sent to the invoker
            f_lambda = None         # compiled numeric function (when needed)
            params: dict = {}       # numeric/extra params for the invoker

            # ---- Per-method input mapping
            if kind in ("bisection", "false_position"):
                # f(x), a, b, tol, max_iter
                expr_text = cleaned.get("fx", "") or ""
                f_lambda, _ = compile_fx(expr_text)
                params = {
                    "a": float(cleaned["a"]),
                    "b": float(cleaned["b"]),
                    "tol": float(cleaned.get("tol", 1e-6)),
                    "max_iter": int(cleaned.get("max_iter", 50)),
                }
                # Plot f on [a,b]
                try:
                    ctx["plot_data_uri"] = _plot_fx_interval(
                        f_lambda, params["a"], params["b"], kind.capitalize()
                    )
                except Exception:
                    pass

            elif kind == "fixed_point":
                # g(x), x0, tol, max_iter
                expr_text = cleaned.get("gx", "") or ""
                f_lambda, _ = compile_fx(expr_text)
                params = {
                    "x0": float(cleaned["x0"]),
                    "tol": float(cleaned.get("tol", 1e-6)),
                    "max_iter": int(cleaned.get("max_iter", 50)),
                    # also pass string name commonly used by FP modules
                    "g_str": expr_text,
                }

            elif kind == "incremental_search":
                # f(x), x0, delta, max_iter
                expr_text = cleaned.get("fx", "") or ""
                f_lambda, _ = compile_fx(expr_text)
                params = {
                    "x0": float(cleaned["x0"]),
                    "delta": float(cleaned.get("delta", 0.5)),
                    "max_iter": int(cleaned.get("max_iter", 50)),
                }

            elif kind == "multiple_roots":
                # Some materials call it h, others f. Accept both.
                expr_text = (
                    cleaned.get("hx")
                    or cleaned.get("fx")
                    or ""
                )
                f_lambda, _ = compile_fx(expr_text)

                df_text = cleaned.get("dhx") or cleaned.get("dfx") or ""
                d2f_text = cleaned.get("d2hx") or cleaned.get("d2fx") or ""
                # (If your module requires strings for df and d2f, pass them too)
                params = {
                    "x0": float(cleaned["x0"]),
                    "tol": float(cleaned.get("tol", 1e-6)),
                    "max_iter": int(cleaned.get("max_iter", 50)),
                }
                if df_text:
                    params["df_str"] = df_text
                if d2f_text:
                    params["d2f_str"] = d2f_text

            elif kind == "newton":
                # f(x), f'(x) (preferred), x0, tol, max_iter
                expr_text = cleaned.get("fx", "") or ""
                f_lambda, _ = compile_fx(expr_text)
                df_text = cleaned.get("dfx", "")  # if the form already asks for it
                params = {
                    "x0": float(cleaned["x0"]),
                    "tol": float(cleaned.get("tol", 1e-6)),
                    "max_iter": int(cleaned.get("max_iter", 50)),
                }
                if df_text:
                    params["df_str"] = df_text  # many modules accept derivative as string

            elif kind == "secant":
                # f(x), x0, x1, tol, max_iter
                expr_text = cleaned.get("fx", "") or ""
                f_lambda, _ = compile_fx(expr_text)
                params = {
                    "x0": float(cleaned["x0"]),
                    "x1": float(cleaned["x1"]),
                    "tol": float(cleaned.get("tol", 1e-6)),
                    "max_iter": int(cleaned.get("max_iter", 50)),
                }

            else:
                ctx["error"] = f"Unsupported roots method: {kind}"
                return render(request, "methods/run_roots.html", ctx)

            # ---- Invoke your real module and capture EXACT console output
            # MUST be (kind, expr_text, f_lambda, params)
            out = invoke_root_algorithm(kind, expr_text, f_lambda, params)
            if out:
                ctx["console"] = out
            else:
                ctx["console"] = "(no output printed by the module)"

        except Exception as e:
            ctx["error"] = str(e)

    return render(request, "methods/run_roots.html", ctx)
