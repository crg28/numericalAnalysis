from sympy import symbols, sympify, lambdify, sin, log
import numpy as np

def falsePosition(a, b, f, tol, max_ite):
    x = symbols('x')
    # le decimos a sympify qué funciones puede usar
    allowed_locals = {"x": x, "sin": sin, "log": log}
    expr = sympify(f, locals=allowed_locals)
    # ahora sí lo convertimos a función numérica con numpy
    f_num = lambdify(x, expr, modules="numpy")

    cont = 1
    ca = b - (f_num(b) * (b - a)) / (f_num(b) - f_num(a))

    print("|Iteration|    xi     |   f(xi)   |     E     |")
    print("| ", 1, " | ", ca, " | ", f_num(ca), " |   |")

    while cont < max_ite:
        cont += 1
        c = ca

        # actualización correcta del intervalo
        if f_num(a) * f_num(c) < 0:
            b = c
        else:
            a = c

        c = b - (f_num(b) * (b - a)) / (f_num(b) - f_num(a))
        eabs = abs(c - ca)
        print("| ", cont, " | ", c, " | ", f_num(c), " | ", eabs, " |")

        if eabs < tol:
            print("Root:", c)
            return c

        ca = c

    print("Limit reached")
    return None

reg_f = falsePosition(0, 1, "log(sin(x)**2 + 1) - 0.5", 1e-7, 100)
