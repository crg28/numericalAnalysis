# vandermonde.py
import numpy as np

def polynomial_string(coefs):
    """
    Build a polynomial string from highest to lowest degree with 6-decimal
    coefficients and compact signs, e.g. -1.141667x^3+5.825000x^2-5.533333x+3.000000
    """
    n = len(coefs)
    parts = []
    for i, c in enumerate(coefs):
        exp = n - 1 - i
        c_rounded = float(np.round(c, 6))
        if abs(c_rounded) < 1e-12:
            continue

        sign = "-" if c_rounded < 0 else "+"
        mag = abs(c_rounded)

        if exp == 0:
            term = f"{mag:.6f}"
        elif exp == 1:
            term = f"{mag:.6f}x"
        else:
            term = f"{mag:.6f}x^{exp}"

        if not parts:
            parts.append(term if sign == "+" else sign + term)
        else:
            parts.append(sign + term)

    return "".join(parts) if parts else "0.000000"

def vandermonde(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    # Build Vandermonde matrix with descending powers: x^(n-1) ... x^0
    V = np.zeros((n, n), dtype=float)
    for r, xv in enumerate(x):
        for c in range(n):
            V[r, c] = xv ** (n - 1 - c)

    # Solve V a = y
    a = np.linalg.solve(V, y)

    # ---- Print results in the required format ----
    print("Vandermonde\n")
    print("Resultados:\n")
    print("Matriz de Vandermonde:\n")
    for row in V:
        print(" " + "  ".join(f"{val: .6f}" for val in row))
    print("\nCoeficientes del polinomio:\n")
    print(" " + "  ".join(f"{coef: .6f}" for coef in a))
    print("\nPolinomio:\n")
    print(polynomial_string(a))
    print()
    return a

if __name__ == "__main__":
    # Data from your table
    x = [-1, 0, 3, 4]
    y = [15.5, 3, 8, 1]

    vandermonde(x, y)
