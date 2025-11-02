import numpy as np

# Try to import matplotlib, but don't crash if not available
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False


def poly_to_string(coeffs):
    """
    Given polynomial coefficients in descending powers,
    e.g. [a3, a2, a1, a0], build a string like:
    -0.050000x^3+0.350000x^2-0.600000x-0.000000
    Always includes all powers down to x^0.
    Uses 6 decimal places.
    """
    n = len(coeffs)
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i - 1  # descending exponent
        # format coeff
        if power == 0:
            term = f"{c:.6f}"
        elif power == 1:
            term = f"{c:.6f}x"
        else:
            term = f"{c:.6f}x^{power}"
        # sign handling: we'll join later, so keep sign on the number
        # but we don't want something like "+-0.5", so we just store raw sign
        terms.append(term)

    # Now we need to join them into a single human-readable expression:
    # First term keeps its sign from formatting
    out = terms[0]
    # Remaining terms: prepend '+' if positive
    for t in terms[1:]:
        if t.strip().startswith('-'):
            out += t  # already has '-'
        else:
            out += "+" + t
    return out


def evaluate_poly(coeffs, x):
    """
    Evaluate polynomial with given descending-power coeffs at scalar x.
    coeffs = [a_n, a_{n-1}, ..., a_0]
    """
    n = len(coeffs)
    total = 0.0
    for i, c in enumerate(coeffs):
        power = n - i - 1
        total += c * (x ** power)
    return total


def build_lagrange_basis(x_points):
    """
    Build the Lagrange basis polynomials L_i(x), i = 0..n-1.

    Returns:
    - basis_polys: list of numpy arrays, each array are coeffs of L_i(x)
                   in descending powers.
                   Each L_i(x) satisfies L_i(x_j) = 1 if j=i else 0.
    """
    x_points = np.array(x_points, dtype=float)
    n = len(x_points)
    basis_polys = []

    for i in range(n):
        # Start with polynomial "1"
        Li = np.array([1.0])
        denom = 1.0

        for j in range(n):
            if j == i:
                continue
            # multiply Li by (x - x_j)
            Li = np.convolve(Li, np.array([1.0, -x_points[j]]))
            # accumulate denominator (x_i - x_j)
            denom *= (x_points[i] - x_points[j])

        # normalize so L_i(x_i) = 1
        Li = Li / denom
        basis_polys.append(Li)

    return basis_polys


def combine_lagrange(basis_polys, y_values):
    """
    Form the full interpolating polynomial:
    P(x) = sum_i y_i * L_i(x)

    Returns:
    - coeffs_final (numpy array), same descending-power format
    """
    n = len(basis_polys)
    # all polys are same length? maybe not, so align by degree
    max_len = max(len(poly) for poly in basis_polys)
    P = np.zeros(max_len)

    for i in range(n):
        Li = basis_polys[i]
        yi = y_values[i]
        # Align Li to max_len (pad on the left)
        Li_aligned = np.zeros(max_len)
        Li_aligned[-len(Li):] = Li
        P += yi * Li_aligned

    return P


def lagrange_interpolation(x_points, y_points, x_eval=None, y_eval=None, show_plot=False):
    """
    Main driver:
    - Builds basis polynomials L_i(x)
    - Builds final polynomial P(x)
    - Optionally evaluates error at (x_eval, y_eval)
    - Optionally plots if matplotlib is available

    Returns dict with:
    - basis_strings: list of strings "poly //Li"
    - combo_string: "y0*L0 + y1*L1 + ..."
    - final_coeffs: numpy array of final polynomial coefficients
    - error_at_eval: |y_eval - P(x_eval)| if provided
    """
    x_points = np.array(x_points, dtype=float)
    y_points = np.array(y_points, dtype=float)
    n = len(x_points)

    # 1. build L_i
    basis_polys = build_lagrange_basis(x_points)

    # 2. convert each L_i to a formatted string
    #    note: poly_to_string expects coeffs in descending powers.
    #    Our Li are already in descending powers from build_lagrange_basis? -> Yes,
    #    because we built via convolve starting from highest power first.
    #    BUT careful: convolution logic produced coeffs increasing power order or descending?
    #    When we convolve with [1, -xj], numpy.convolve([1],[1,-xj]) gives [1,-xj].
    #    That is descending powers already? Let's check:
    #    Polynomial [a,b,c] we interpret as a*x^2 + b*x + c. That's descending powers.
    #    After np.convolve(Li, [1, -xj]) Li grows with new highest power in index 0.
    #    So yes: Li[0] multiplies x^(deg), Li[-1] is constant. So descending order is correct.

    basis_strings = []
    for idx, Li in enumerate(basis_polys):
        sLi = poly_to_string(Li)
        basis_strings.append((idx, sLi))

    # 3. final polynomial coefficients
    final_coeffs = combine_lagrange(basis_polys, y_points)

    # 4. combination string like "15.5*L0+3*L1+8*L2+L3"
    combo_terms = []
    for i, yi in enumerate(y_points):
        # if coefficient is exactly 1.0 or very close -> just "L<i>"
        if abs(yi - 1.0) < 1e-12:
            combo_terms.append(f"L{i}")
        else:
            combo_terms.append(f"{yi}*L{i}")
    combo_string = "+".join(combo_terms)

    # 5. compute error at eval point if provided
    error_at_eval = None
    if x_eval is not None and y_eval is not None:
        p_at_x_eval = evaluate_poly(final_coeffs, x_eval)
        error_at_eval = abs(y_eval - p_at_x_eval)

    # 6. plot if asked
    if show_plot and _HAS_MPL:
        # dense sampling for the curve
        xmin = np.min(x_points)
        xmax = np.max(x_points)
        x_plot = np.linspace(xmin - 1, xmax + 1, 200)
        y_plot = [evaluate_poly(final_coeffs, xx) for xx in x_plot]

        plt.figure(figsize=(7,5))
        plt.plot(x_plot, y_plot, label="Interpolating polynomial (Lagrange)")
        plt.scatter(x_points, y_points, c="red", label="Data points", zorder=5)

        if x_eval is not None and y_eval is not None:
            plt.scatter([x_eval], [y_eval], c="green", s=80, label="True point")
            plt.scatter([x_eval], [p_at_x_eval], c="orange", s=80, label="Interpolated")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Lagrange interpolation")
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        "basis_strings": basis_strings,
        "combo_string": combo_string,
        "final_coeffs": final_coeffs,
        "error_at_eval": error_at_eval,
    }


def print_lagrange_report(x_points, y_points, result):
    """
    Prints in the style you showed:
    
    Lagrange

    Results:

    Lagrange interpolating polynomials:
        <poly for L0>   //L0
        <poly for L1>   //L1
        ...
    
    Polynomial:
        y0*L0 + y1*L1 + ...
    """
    print("\nLagrange\n")
    print("Results:\n")
    print("Lagrange interpolating polynomials:")

    for idx, poly_str in result["basis_strings"]:
        # align with your screenshot: polynomial first, then //L<i>
        print(f"  {poly_str:<60}  //L{idx}")

    print("\nPolynomial:\n")
    print(result["combo_string"])
    print()

    # Optional: also show final coefficients of P(x)
    coeffs = result["final_coeffs"]
    full_poly_str = poly_to_string(coeffs)
    print("\nExpanded P(x):")
    print(full_poly_str)

    if result["error_at_eval"] is not None:
        print(f"\nAbsolute interpolation error at eval point: {result['error_at_eval']:.6e}")


if __name__ == "__main__":
    # Example based on your screenshot style:
    # Table:
    # x: -1    0    3    4
    # y: 15.5  3    8    1
    x_data = [-1, 0, 3, 4]
    y_data = [15.5, 3, 8, 1]

    # You can also pass a point to test interpolation accuracy
    # (not shown in screenshot, but supported)
    x_eval = None
    y_eval = None

    result = lagrange_interpolation(
        x_points=x_data,
        y_points=y_data,
        x_eval=x_eval,
        y_eval=y_eval,
        show_plot=False  # keep CLI output only
    )

    print_lagrange_report(x_data, y_data, result)
