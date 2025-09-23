from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("bisection/", views.bisection_page, name="bisection"),
    path("false-position/", views.false_position_page, name="false_position"),
    path("fixed-point/", views.fixed_point_page, name="fixed_point"),
    path("gaussian-elimination/", views.gaussian_elimination_page, name="gaussian_elimination"),
    path("newton/", views.newton_page, name="newton"),
    path("partial-pivoting/", views.partial_pivoting_page, name="partial_pivoting"),
    path("total-pivoting/", views.total_pivoting_page, name="total_pivoting"),
    path("roots-multiplicity/", views.roots_multiplicity_page, name="multiplicity_page"),
    path("secant/", views.secant_page, name="secant"),


]