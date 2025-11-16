from django.urls import path
from .views import views_core as core
from .views import views_linear as linear
from .views import views_roots as roots
from .views import views_interp as interp

app_name = "methods"
urlpatterns = [
    path("", core.home, name="home"),
    path("methods/", core.method_list, name="method_list"),
    path("category/<slug:slug>/", core.category_view, name="category"),

    # Roots (genérico por slug)
    path("roots/<slug:slug>/", roots.method_run_roots, name="roots_run"),

    # Linear (si usas una sola vista con slug)
    path("linear/<slug:slug>/", linear.method_run_linear, name="linear_run"),

    # Interp (cuando lo tengas)
    path("interp/<slug:slug>/", interp.method_run_interp, name="interp_run"),
    path("search/", core.method_list, name="search_products"),
    path("m/<slug:slug>/", core.legacy_method_entrypoint, name="detail"),  # ← alias para 'methods:detail'
    path("list/", core.method_list, name="list"),         
]
# methods/urls.py
from django.urls import path
from .views import views_core as core
from .views import views_linear as linear
from .views import views_roots as roots
from .views import views_interp as interp  # keep even if interp_run is commented for now

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
]
