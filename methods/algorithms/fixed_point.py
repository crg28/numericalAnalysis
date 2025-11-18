# methods/algorithms/fixed_point.py

from sympy import symbols, sympify, lambdify

# ---------------------------
# Core of the method
# ---------------------------
def fixed_point(f, g, x0, tol=1e-7, max_iter=30):
    """
    Fixed-Point Method.
    f(x) = g(x) - x
    Receives two numerical functions f and g, prints the iteration table,
    and returns (xn, iter).
    """

    print("{:<6} {:<15} {:<15} {:<15} {:<15}".format("iter", "xi", "g(xi)", "f(xi)", "E"))
    print("-" * 70)

    xn = float(x0)

    for i in range(max_iter + 1):
        gx = float(g(xn))
        fx = float(f(xn))

        if i == 0:
            # first row without error
            print("{:<6} {:<15.8f} {:<15.8f} {:<15.2e} {:<15}".format(i, xn, gx, fx, ""))
        else:
            error = abs(gx - xn)
            print("{:<6} {:<15.8f} {:<15.8f} {:<15.2e} {:<15.2e}".format(i, xn, gx, fx, error))

            if error < tol:
                print(f"\nAn approximation of the root was found at {xn:.15f}")
                return xn, i

        xn = gx  # next iteration

    print("\nTolerance was not reached within the maximum number of iterations")
    return xn, max_iter


# ---------------------------
# Wrapper to integrate with Django
# ---------------------------
_x = symbols("x")

def run(g_str, x0, tol=1e-7, max_iter=30):
    """
    Function used by the generic invoker.
    - g_str: string for g(x) (e.g., 'log(sin(x)**2 + 1) - 1/2')
    - x0: initial value
    - tol, max_iter: stopping parameters
    Builds numerical g(x) from the string and defines f(x) = g(x) - x.
    """

    # Numerical g(x) using sympy + numpy
    g = lambdify(_x, sympify(g_str, convert_xor=True), "numpy")
    f = lambda t: g(t) - t

    return fixed_point(f, g, x0, tol=tol, max_iter=max_iter)


# ---------------------------
# Optional test block
# ---------------------------
if __name__ == "__main__":
    # Quick example: same as before
    example_g = "log(sin(x)**2 + 1) - 1/2"
    run(example_g, x0=-0.5, tol=1e-7, max_iter=30)
