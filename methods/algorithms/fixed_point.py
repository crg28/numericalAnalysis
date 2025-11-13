# methods/algorithms/fixed_point.py

from sympy import symbols, sympify, lambdify

# ---------------------------
# Núcleo del método
# ---------------------------
def fixed_point(f, g, x0, tol=1e-7, max_iter=30):
    """
    Método de Punto Fijo.
    f(x) = g(x) - x
    Recibe dos funciones numéricas f y g, imprime la tabla y devuelve (xn, iter).
    """

    print("{:<6} {:<15} {:<15} {:<15} {:<15}".format("iter", "xi", "g(xi)", "f(xi)", "E"))
    print("-" * 70)

    xn = float(x0)

    for i in range(max_iter + 1):
        gx = float(g(xn))
        fx = float(f(xn))

        if i == 0:
            # primera fila sin error
            print("{:<6} {:<15.8f} {:<15.8f} {:<15.2e} {:<15}".format(i, xn, gx, fx, ""))
        else:
            error = abs(gx - xn)
            print("{:<6} {:<15.8f} {:<15.8f} {:<15.2e} {:<15.2e}".format(i, xn, gx, fx, error))

            if error < tol:
                print(f"\nAn approximation of the root was found at {xn:.15f}")
                return xn, i

        xn = gx  # siguiente iteración

    print("\nTolerance was not reached within the maximum number of iterations")
    return xn, max_iter


# ---------------------------
# Wrapper para integrarse con Django
# ---------------------------
_x = symbols("x")

def run(g_str, x0, tol=1e-7, max_iter=30):
    """
    Función que usa el invocador genérico.
    - g_str: cadena con g(x) (por ej. 'log(sin(x)**2 + 1) - 1/2')
    - x0: valor inicial
    - tol, max_iter: parámetros de parada
    Construye g(x) numérica a partir del string y define f(x) = g(x) - x.
    """

    # g(x) numérica usando sympy + numpy
    g = lambdify(_x, sympify(g_str, convert_xor=True), "numpy")
    f = lambda t: g(t) - t

    return fixed_point(f, g, x0, tol=tol, max_iter=max_iter)


# ---------------------------
# Bloque de prueba opcional
# ---------------------------
if __name__ == "__main__":
    # Ejemplo rápido: mismo que tenías antes
    ejemplo_g = "log(sin(x)**2 + 1) - 1/2"
    run(ejemplo_g, x0=-0.5, tol=1e-7, max_iter=30)
