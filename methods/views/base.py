# methods/views/base.py
from __future__ import annotations
from django.contrib.auth import login
from django.contrib.auth.views import LoginView, LogoutView
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

from ..forms import SignUpForm
from ..models import Category, Method


# ================================================================
# Utilidades de contexto
# ================================================================
def _common_ctx():
    """Retorna el contexto base con las categorías ordenadas."""
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
    """Redirige a la vista de lista con filtro de búsqueda."""
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
