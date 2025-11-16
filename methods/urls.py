# methods/urls.py
from django.urls import path
from .views import views_core as core
from .views import views_linear as linear
from .views import views_roots as roots
from .views import views_interp as interp  # aunque interp no se use todavía
from .views import views_graph

app_name = "methods"

urlpatterns = [
    path("", core.home, name="home"),
    path("methods/", core.method_list, name="method_list"),
    path("category/<slug:slug>/", core.category_view, name="category"),

    # Roots (generic by slug)
    path("roots/<slug:slug>/", roots.method_run_roots, name="roots_run"),

    # Linear (generic by slug)
    path("linear/<slug:slug>/", linear.method_run_linear, name="linear_run"),

    # Interpolation (enable when ready)
    path("interp/<slug:slug>/", interp.method_run_interp, name="interp_run"),

    # Search/list aliases for templates
    path("search/", core.method_list, name="search_products"),
    path("m/<slug:slug>/", core.legacy_method_entrypoint, name="detail"),
    path("list/", core.method_list, name="list"),

    # 🔹 Function plotter
    path("graph/", views_graph.function_plotter, name="function_plotter"),
]
