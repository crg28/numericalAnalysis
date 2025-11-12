# methods/views/views_core.py
from __future__ import annotations
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from ..models import Category, Method


def _common_ctx():
    return {"categories": Category.objects.all().order_by("name")}


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


# Map method.kind → group (support ES/EN slugs)
_KIND_GROUP = {
    # Linear systems
    "jacobi": "linear",
    "gauss_seidel": "linear",
    "sor": "linear",
    "gaussian_elimination": "linear",
    "pivot_partial": "linear",
    "pivot_total": "linear",
    "crout": "linear",
    "doolittle": "linear",
    "cholesky": "linear",
    "lu_simple": "linear",
    "lu_pivot": "linear",

    # Roots (ES/EN)
    "biseccion": "roots",
    "bisection": "roots",
    "regla_falsa": "roots",
    "false_position": "roots",
    "busqueda_incremental": "roots",
    "incremental_search": "roots",
    "newton": "roots",
    "secante": "roots",
    "secant": "roots",
    "punto_fijo": "roots",
    "fixed_point": "roots",
    "raices_multiples": "roots",
    "multiple_roots": "roots",

    # Interpolation (only if you enable its URL)
    "lagrange": "interp",
    "vandermonde": "interp",
    "newton_interpolation": "interp",
    "spline": "interp",
    "spline_cubic": "interp",
    "spline_quadratic": "interp",
    "spline_linear": "interp",
}


def legacy_method_entrypoint(request, slug):
    """
    Receives /m/<slug>/ from templates and redirects to
    /roots/<slug>/, /linear/<slug>/, or the list page.
    """
    method = get_object_or_404(Method, slug=slug)
    group = _KIND_GROUP.get(method.kind, "linear")

    if group == "roots":
        url = reverse("methods:roots_run", kwargs={"slug": slug})
    elif group == "linear":
        url = reverse("methods:linear_run", kwargs={"slug": slug})
    elif group == "interp":
        # If interpolation URL is not enabled yet, fallback to list.
        try:
            url = reverse("methods:interp_run", kwargs={"slug": slug})
        except Exception:
            url = reverse("methods:method_list")
    else:
        url = reverse("methods:method_list")

    return redirect(url)
