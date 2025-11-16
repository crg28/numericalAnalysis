from django.http import HttpResponse
from .utils_interp import run_linear_spline,run_quadratic_spline,run_cubic_spline, run_lagrange, run_vandermonde, run_newton_interpolation

def method_run_interp(request, slug):

    if slug == "spline-linear":
        return run_linear_spline(request,slug)

    if slug == "spline-quadratic":
        return run_quadratic_spline(request,slug)
    
    if slug == "spline":
        return run_cubic_spline(request,slug)
    
    if slug == "lagrange":
        return run_lagrange(request,slug) 
    
    if slug == "vandermonde":
        return run_vandermonde(request,slug)
    
    if slug == "newton-interpolation":
        return run_newton_interpolation(request,slug)



    















































""" # methods/views/views_interp.py
from __future__ import annotations
from django.shortcuts import render, get_object_or_404
from django import forms
from ..models import Method


class InterpForm(forms.Form):
    xs = forms.CharField(label="x data (comma/space/lines)", help_text="e.g., 0,1,2,3")
    ys = forms.CharField(label="y data (comma/space/lines)", help_text="e.g., 1,2,0,4")


def _parse_vector(s: str):
    t = (s or "").strip()
    if not t:
        return []
    if t.startswith("["):
        from ast import literal_eval
        return [float(v) for v in literal_eval(t)]
    return [float(p) for p in t.replace(",", " ").split() if p]


def method_run_interp(request, slug):
    method = get_object_or_404(Method, slug=slug)
    form = InterpForm(request.POST or None)
    ctx = {"method": method, "form": form, "poly": None, "error": None}

    if request.method == "POST" and form.is_valid():
        try:
            xs = _parse_vector(form.cleaned_data["xs"])
            ys = _parse_vector(form.cleaned_data["ys"])
            if len(xs) != len(ys) or len(xs) < 2:
                ctx["error"] = "x and y must have the same length (n ≥ 2)."
            else:
                # Skeleton: show loaded points. Hook your algorithms here.
                ctx["poly"] = f"Loaded points: {list(zip(xs, ys))}"
        except Exception as e:
            ctx["error"] = str(e)

    return render(request, "methods/run_interp.html", ctx)
 """