from django.shortcuts import render

def home(request):
    return render(request, "methods/home.html")

def bisection_page(request):
    return render(request, "methods/bisection.html")

def false_position_page(request):
    return render(request, "methods/false_position.html")

def fixed_point_page(request):
    return render(request, "methods/fixed_point.html")

def gaussian_elimination_page(request):
    return render(request, "methods/gaussian_elimination.html")

def newton_page(request):
    return render(request, "methods/newton.html")

def partial_pivoting_page(request):
    return render(request, "methods/partial_pivoting.html")

def total_pivoting_page(request):
    return render(request, "methods/total_pivoting.html")

def roots_multiplicity_page(request):
    return render(request, "methods/roots_multiplicity.html")

def secant_page(request):
    return render(request, "methods/secant.html")


