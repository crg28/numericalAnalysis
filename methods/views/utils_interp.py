from django.shortcuts import render
from methods.algorithms.spline import spline
from methods.algorithms.lagrange import lagrange,poly_line_full
from methods.algorithms.vandermonde import vandermonde,polynomial_string
from methods.algorithms.newton_int import newtonint
from .forms_interp import SplineBaseForm, LagrangeForm
import numpy as np

# ==========================
#   AUXILIAR: FORMATEAR POLIS
# ==========================
def convert_spline_table_to_text(table, degree):
    lines = []

    for row in table:
        poly = ""
        for j, coef in enumerate(row):
            exp = degree - j   

            if abs(coef) < 1e-12 and exp != 0:
                continue

            if poly == "":
                term = f"{coef:.6f}"
            else:
                term = f" + {coef:.6f}" if coef >= 0 else f" - {abs(coef):.6f}"

            if exp > 0:
                term += "x"
            if exp > 1:
                term += f"^{exp}"

            poly += term

        lines.append(poly)

    return "\n".join(lines)


def run_linear_spline(request, slug):

    if request.method == "GET":
        form = SplineBaseForm()
        return render(request, "methods/run_interp.html", {
            "form": form,
            "method_name": "Spline Lineal",
            "poly": None,
            "eval_result": None,
            "error": None,
        })

    if request.method == "POST":
        form = SplineBaseForm(request.POST)

        if not form.is_valid():
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Spline Lineal",
                "poly": None,
                "eval_result": None,
                "error": "Datos inválidos. Verifique X y Y.",
            })

        x = form.cleaned_data["x_list"]
        y = form.cleaned_data["y_list"]
        x_eval = form.cleaned_data["x_point"]

        try:
            # -------- CALCULAR SPLINE --------
            table = spline(x, y, 1)

            # -------- TEXTO POLINOMIOS --------
            poly = convert_spline_table_to_text(table, degree=1)

            # -------- EVALUAR EL SPLINE --------
            eval_result = None

            if x_eval is not None:
                for i in range(len(x) - 1):

                    if x[i] <= x_eval <= x[i+1]:

                        a, b = table[i]

                        # CORRECTO: polinomio forma ax + b
                        eval_result = a * x_eval + b
                        break

                if eval_result is None:
                    eval_result = "Fuera del dominio del spline."

            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Spline Lineal",
                "poly": poly,
                "eval_result": eval_result,
                "error": None,
            })

        except Exception as e:
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Spline Lineal",
                "poly": None,
                "eval_result": None,
                "error": f"Error al calcular el spline: {e}",
            })

def newton_to_standard(coef, x):
    """
    Convierte los coeficientes de Newton a coeficientes estándar (forma ax^n + ... + b).
    """
    n = len(coef)
    poly = np.array([1.0])  # polinomio 1

    result = coef[0] * poly  # empieza con c0

    for i in range(1, n):
        # multiplicar poly por (x - xi)
        poly = np.convolve(poly, [1, -x[i-1]])
        result = np.pad(result, (0, len(poly) - len(result)))
        result = result + coef[i] * poly
    
    return result






def run_quadratic_spline(request, slug):

    if request.method == "GET":
        form = SplineBaseForm()
        return render(request, "methods/run_interp.html", {
            "form": form,
            "method_name": "Spline Cuadrático",
            "poly": None,
            "eval_result": None,
            "error": None,
        })

    if request.method == "POST":
        form = SplineBaseForm(request.POST)

        if not form.is_valid():
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Spline Cuadrático",
                "poly": None,
                "eval_result": None,
                "error": "Datos inválidos. Verifique X y Y.",
            })

        x = form.cleaned_data["x_list"]
        y = form.cleaned_data["y_list"]
        x_eval = form.cleaned_data["x_point"]

        try:
            # -------- CALCULAR SPLINE CUADRÁTICO --------
            table = spline(x, y, 2)

            # -------- TEXTO POLINOMIOS --------
            poly = convert_spline_table_to_text(table, degree=2)

            # -------- EVALUAR EL SPLINE CUADRÁTICO --------
            eval_result = None

            if x_eval is not None:
                for i in range(len(x) - 1):

                    if x[i] <= x_eval <= x[i+1]:

                        a, b, c = table[i]  # ax² + bx + c
                        eval_result = a * x_eval**2 + b * x_eval + c
                        break

                if eval_result is None:
                    eval_result = "Fuera del dominio del spline."

            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Spline Cuadrático",
                "poly": poly,
                "eval_result": eval_result,
                "error": None,
            })

        except Exception as e:
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Spline Cuadrático",
                "poly": None,
                "eval_result": None,
                "error": f"Error al calcular el spline: {e}",
            })




def run_cubic_spline(request, slug):

    if request.method == "GET":
        form = SplineBaseForm()
        return render(request, "methods/run_interp.html", {
            "form": form,
            "method_name": "Spline Cúbico",
            "poly": None,
            "eval_result": None,
            "error": None,
        })

    if request.method == "POST":
        form = SplineBaseForm(request.POST)

        if not form.is_valid():
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Spline Cúbico",
                "poly": None,
                "eval_result": None,
                "error": "Datos inválidos. Verifique X y Y.",
            })

        x = form.cleaned_data["x_list"]
        y = form.cleaned_data["y_list"]
        x_eval = form.cleaned_data["x_point"]

        try:
            # -------- CALCULAR SPLINE CÚBICO --------
            table = spline(x, y, 3)

            # -------- TEXTO POLINOMIOS --------
            poly = convert_spline_table_to_text(table, degree=3)

            # -------- EVALUAR EL SPLINE CÚBICO --------
            eval_result = None

            if x_eval is not None:
                for i in range(len(x) - 1):

                    if x[i] <= x_eval <= x[i+1]:

                        a, b, c, d = table[i]  # ax³ + bx² + cx + d
                        eval_result = (
                            a * x_eval**3 +
                            b * x_eval**2 +
                            c * x_eval +
                            d
                        )
                        break

                if eval_result is None:
                    eval_result = "Fuera del dominio del spline."

            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Spline Cúbico",
                "poly": poly,
                "eval_result": eval_result,
                "error": None,
            })

        except Exception as e:
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Spline Cúbico",
                "poly": None,
                "eval_result": None,
                "error": f"Error al calcular el spline: {e}",
            })




def run_lagrange(request, slug):

    if request.method == "GET":
        form = LagrangeForm()
        return render(request, "methods/run_interp.html", {
            "form": form,
            "method_name": "Interpolación de Lagrange",
            "poly": None,
            "eval_result": None,
            "error": None,
        })

    if request.method == "POST":
        form = LagrangeForm(request.POST)

        if not form.is_valid():
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Interpolación de Lagrange",
                "poly": None,
                "eval_result": None,
                "error": "Datos inválidos. Verifique X y Y.",
            })

        x = form.cleaned_data["x_values"]
        y = form.cleaned_data["y_values"]
        x_eval = form.cleaned_data["x_eval"]


        try:
            # ==========================
            #   CALCULAR L_i(x)
            # ==========================
            L_table = lagrange(x, y)    # solo usa tu método, no mostramos la tabla

            # ==========================
            #   CONSTRUIR POLINOMIO FINAL
            #   P(x) = sum( y_i * L_i(x) )
            # ==========================
            n = len(x)
            P = np.zeros(n)

            for i in range(n):
                P += y[i] * L_table[i]

            poly_text = poly_line_full(P)

            # ==========================
            #       EVALUACIÓN
            # ==========================
            eval_result = None
            if x_eval is not None:
                eval_result = float(np.polyval(P, x_eval))

            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Interpolación de Lagrange",
                "poly": poly_text,
                "eval_result": eval_result,
                "error": None,
            })

        except Exception as e:
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Interpolación de Lagrange",
                "poly": None,
                "eval_result": None,
                "error": f"Error al calcular Lagrange: {e}",
            })



""" def run_newton_interpolation(request, slug):
    if request.method == "GET":
        form = LagrangeForm()
        return render(request, "methods/run_interp.html", {
            "form": form,
            "method_name": "Interpolación de Newton",
            "poly": None,
            "eval_result": None,
            "diff_table": None,
            "error": None,
        })

    if request.method == "POST":
        form = LagrangeForm(request.POST)

        if not form.is_valid():
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Interpolación de Newton",
                "poly": None,
                "eval_result": None,
                "diff_table": None,
                "error": "Datos inválidos. Verifique X y Y.",
            })

        x = form.cleaned_data["x_values"]
        y = form.cleaned_data["y_values"]
        x_eval = form.cleaned_data["x_eval"]

        # Validación de longitud
        if len(x) != len(y):
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Interpolación de Newton",
                "poly": None,
                "eval_result": None,
                "diff_table": None,
                "error": "Las listas X y Y deben tener la misma cantidad de elementos.",
            })

        try:
            # ==========================
            #   CALCULAR POLINOMIO
            # ==========================
            poly_text, coef, diff_table = newtonint(x, y)

            # ==========================
            #      EVALUACIÓN
            # ==========================
            eval_result = None
            if x_eval is not None:
                # Conversión correcta a polinomio estándar
                poly_standard = newton_to_standard(coef, x)
                eval_result = float(np.polyval(poly_standard, x_eval))

            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Interpolación de Newton",
                "poly": poly_text,
                "eval_result": eval_result,
                "diff_table": diff_table,
                "error": None,
            })

        except Exception as e:
            print(e)
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Interpolación de Newton",
                "poly": None,
                "eval_result": None,
                "diff_table": None,
                "error": f"Error en Newton: {e}",
            }) """




# ==============================
#   Evaluar polinomio de Newton
# ==============================
def evaluate_newton_form(coef, x_nodes, x_value):
    n = len(coef)
    result = coef[-1]
    # Horner modificado para la forma de Newton
    for i in range(n - 2, -1, -1):
        result = result * (x_value - x_nodes[i]) + coef[i]
    return result


def run_newton_interpolation(request, slug):
    if request.method == "GET":
        form = LagrangeForm()
        return render(request, "methods/run_interp.html", {
            "form": form,
            "method_name": "Interpolación de Newton (DEBUG)",
            "poly": None,
            "eval_result": None,
            "diff_table": None,
            "error": None,
        })

    if request.method == "POST":
        form = LagrangeForm(request.POST)

        if not form.is_valid():
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Interpolación de Newton (DEBUG)",
                "poly": None,
                "eval_result": None,
                "diff_table": None,
                "error": "Datos inválidos. Verifique X y Y.",
            })

        x = form.cleaned_data["x_values"]
        y = form.cleaned_data["y_values"]
        x_eval = form.cleaned_data["x_eval"]

        # Validación de longitud
        if len(x) != len(y):
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Interpolación de Newton (DEBUG)",
                "poly": None,
                "eval_result": None,
                "diff_table": None,
                "error": "Las listas X y Y deben tener la misma cantidad de elementos.",
            })

        # ==========================
        #   CALCULAR POLINOMIO
        # ==========================
        poly_text, coef, diff_table = newtonint(x, y)

        # Convertir tabla a lista para el template
        diff_table = diff_table.tolist()

        # ==========================
        #      EVALUACIÓN CORRECTA
        # ==========================
        eval_result = None
        if x_eval is not None:
            eval_result = evaluate_newton_form(coef, x, x_eval)

        return render(request, "methods/run_interp.html", {
            "form": form,
            "method_name": "Interpolación de Newton (DEBUG)",
            "poly": poly_text,
            "eval_result": eval_result,
            "diff_table": diff_table,
            "error": None,
        })






def run_vandermonde(request, slug):

    if request.method == "GET":
        form = LagrangeForm()
        return render(request, "methods/run_interp.html", {
            "form": form,
            "method_name": "Método de Vandermonde",
            "poly": None,
            "eval_result": None,
            "error": None,
        })

    if request.method == "POST":
        form = LagrangeForm(request.POST)

        if not form.is_valid():
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Método de Vandermonde",
                "poly": None,
                "eval_result": None,
                "error": "Datos inválidos. Verifique X y Y.",
            })

        x = form.cleaned_data["x_values"]
        y = form.cleaned_data["y_values"]
        x_eval = form.cleaned_data["x_eval"]

        try:
            # ==========================
            #       CALCULAR POLINOMIO
            # ==========================
            a = vandermonde(x, y)   # tu función ya retorna los coeficientes

            # texto del polinomio
            poly_text = polynomial_string(a)

            # ==========================
            #       EVALUACIÓN
            # ==========================
            eval_result = None
            if x_eval is not None:
                eval_result = float(np.polyval(a, x_eval))

            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Método de Vandermonde",
                "poly": poly_text,
                "eval_result": eval_result,
                "error": None,
            })

        except Exception as e:
            return render(request, "methods/run_interp.html", {
                "form": form,
                "method_name": "Método de Vandermonde",
                "poly": None,
                "eval_result": None,
                "error": f"Error en Vandermonde: {e}",
            })