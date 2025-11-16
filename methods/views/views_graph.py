# methods/views/views_graph.py
from django.shortcuts import render
from ..forms import GraphForm  # si no quieres usar el form, podemos omitir

def function_plotter(request):
    """
    Independent function plotter page (GeoGebra-style).
    All plotting is done client-side with math.js + Plotly.
    """
    return render(request, "methods/function_plotter.html", {})
