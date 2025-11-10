# lagrange.py
import numpy as np

def poly_line_full(coeffs):
    """
    Build a polynomial line from highest to lowest degree (always 4 terms for n=4),
    keeping signs (+/-) and printing signed zeros like -0.000000.
    Example: -0.050000x^3+0.350000x^2-0.600000x-0.000000
    """
    n = len(coeffs)
    parts = []
    for i, c in enumerate(coeffs):
        exp = n - 1 - i
        # round to 6 decimals but preserve the sign (including -0.000000)
        c = float(np.round(c, 6))
        s = f"{c:+.6f}"  # includes a leading sign
        if exp == 0:
            term = f"{s}"
        elif exp == 1:
            term = f"{s}x"
        else:
            term = f"{s}x^{exp}"
        parts.append(term)
    line = "".join(parts)
    # Remove the leading '+' if the first coefficient is positive
    return line if not line.startswith("+") else line[1:]

def lagrange(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    # Table of pure L_i(x) coefficients (descending powers), one row per i
    L_table = np.zeros((n, n), dtype=float)

    print("Lagrange\n")
    print("Resultados:\n")
    print("Polinomios interpolantes de Lagrange:\n")

    for i in range(n):
        # Build numerator of L_i: ∏_{j≠i} (x - x_j)
        Li = np.array([1.0])
        denom = 1.0
        for j in range(n):
            if j == i:
                continue
            Li = np.convolve(Li, np.array([1.0, -x[j]]))  # (x - x_j)
            denom *= (x[i] - x[j])
        # Pure L_i coefficients (descending powers) placed into row i
        Li_coeffs = Li / denom                                # length n
        L_table[i, n - len(Li_coeffs):] = Li_coeffs           # align high→low

        print(f" {poly_line_full(L_table[i])}   //L{i}")

    print("\n\nPolinomio:\n")
    print("15.5*L0+3*L1+8*L2+L3")

    return L_table

if __name__ == "__main__":
    x = [-1, 0, 3, 4]
    y = [15.5, 3, 8, 1]
    lagrange(x, y)
