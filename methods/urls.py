from django.urls import path
from . import views

app_name = "methods"

urlpatterns = [
    path("", views.home, name="home"),
    path("methods/", views.method_list, name="list"),          
    path("m/<slug:slug>/", views.method_run, name="detail"),
    path("category/<slug:slug>/", views.category_view, name="category"),
    path("search/", views.search_products, name="search_products"),
]
