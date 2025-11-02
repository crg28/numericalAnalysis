import numpy as np

# Try to import matplotlib, but don't crash if it's not available
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
    Always keeps all terms down to x^0.
    """
    n = len(coeffs)
    terms = []
    for i, c in enumerate(coeffs):
        p = n - i - 1  # exponent
        if p == 0:
            term = f"{c:.6f}"
        elif p == 1:
            term = f"{c:.6f}x"
        else:
            term = f"{c:.6f}x^{p}"
        terms.append(term)

    # Join terms with +/-
    out = terms[0]
    for t in terms[1:]:
        if t.strip().startswith('-'):
            out += t
        else:
            out += "+" + t
    return out


def evaluate_poly(coeffs, x):
    """
    Evaluate polynomial with descending-power coeffs at scalar x.
    coeffs = [a_n, a_{n-1}, ..., a_0]
    """
    total = 0.0
    n = len(coeffs)
    for i, c in enumerate(coeffs):
        p = n - i - 1
        total += c * (x ** p)
    return total


def build_vandermonde_matrix(x_vals):
    """
    Build the Vandermonde matrix A where:
    A[i,j] = x_i^(n-1-j)
    So columns go x^(n-1), x^(n-2), ..., x^1, x^0
    """
    x_vals = np.array(x_vals, dtype=float)
    n = len(x_vals)
    A = np.vander(x_vals, N=n, increasing=False)
    return A


def vandermonde_interpolation(x_vals, y_vals, x_eval=None, y_eval=None):
    """
    Perform polynomial interpolation using Vandermonde.

    Steps:
    - Build Vandermonde matrix A
    - Solve A * a = y to get polynomial coefficients a
      in descending powers
    - Build pretty string for P(x)
    - Optionally compute interpolation error at (x_eval, y_eval)

    Returns:
    - A: Vandermonde matrix
    - coeffs: polynomial coefficients [a_n,...,a_0]
    - poly_str: human-readable polynomial string
    - y_eval_est: P(x_eval), if x_eval given
    - abs_error: |y_eval - P(x_eval)|, if x_eval & y_eval given
    """
    x_vals = np.array(x_vals, dtype=float)
    y_vals = np.array(y_vals, dtype=float)

    if len(x_vals) != len(y_vals):
        raise ValueError("x and y must have the same length.")

    # 1. Build Vandermonde matrix
    A = build_vandermonde_matrix(x_vals)

    # 2. Solve the linear system A * a = y
    #    coeffs = [a_n, a_{n-1}, ..., a_0]
    coeffs = np.linalg.solve(A, y_vals)

    # 3. Pretty polynomial text
    poly_str = poly_to_string(coeffs)

    # 4. Error at evaluation point
    y_eval_est = None
    abs_error = None
    if x_eval is not None and y_eval is not None:
        y_eval_est = evaluate_poly(coeffs, x_eval)
        abs_error = abs(y_eval - y_eval_est)

    return {
        "A": A,
        "coeffs": coeffs,
        "poly_str": poly_str,
        "y_eval_est": y_eval_est,
        "abs_error": abs_error,
    }


def maybe_plot(x_vals, y_vals, coeffs, x_eval=None, y_eval=None):
    """
    Plot data points and polynomial curve if matplotlib is available.
    """
    if not _HAS_MPL:
        return  # silently skip

    x_vals = np.array(x_vals, dtype=float)
    xmin = np.min(x_vals)
    xmax = np.max(x_vals)

    xp = np.linspace(xmin - 1, xmax + 1, 200)
    yp = [evaluate_poly(coeffs, xx) for xx in xp]

    plt.figure(figsize=(7,5))
    plt.plot(xp, yp, label="Interpolating polynomial (Vandermonde)")
    plt.scatter(x_vals, y_vals, c="red", zorder=5, label="Data points")

    if x_eval is not None and y_eval is not None:
        plt.scatter([x_eval], [y_eval], c="green", s=80, label="True point")
        plt.scatter([x_eval], [evaluate_poly(coeffs, x_eval)],
                    c="orange", s=80, label="Interpolated")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Vandermonde interpolation")
    plt.grid(True)
    plt.legend()
    plt.show()


def print_vandermonde_report(x_vals, y_vals, x_eval, y_eval, result):
    """
    Prints a report similar in style to what you want:
    
    Vandermonde

    Results:

    Vandermonde matrix A:
    [ ... ]
    [ ... ]

    Coefficients (highest power first):
    [a_n ... a_0]

    Interpolating polynomial:
    P(x) = ...

    If evaluation point provided:
    P(x_eval) = ...
    abs error = ...
    """
    print("\nVandermonde\n")
    print("Results:\n")

    # Matrix A
    print("Vandermonde matrix A:")
    A = result["A"]
    for row in A:
        print("  " + "  ".join(f"{val: .6f}" for val in row))
    print()

    # Coefficients
    print("Coefficients (highest power first):")
    # show as row vector style
    coeff_list = "  " + "  ".join(f"{c:.6f}" for c in result["coeffs"])
    print(coeff_list)
    print()

    # Polynomial
    print("Interpolating polynomial:")
    print("P(x) =", result["poly_str"])
    print()

    # Optional error info
    if result["y_eval_est"] is not None and result["abs_error"] is not None:
        print(f"P({x_eval}) = {result['y_eval_est']:.6f}")
        print(f"Absolute error |f(x_eval) - P(x_eval)| = {result['abs_error']:.6f}")
        print()


def read_user_input():
    """
    Asks the user for:
    - data table (x and y)
    - optional evaluation point
    - whether to plot
    """
    print("Vandermonde interpolation")
    print("You will enter the data points (x_i, y_i).")
    n = int(input("How many data points? "))

    x_vals = []
    y_vals = []
    print("\nEnter the data table.")
    print("For each point, type: x y")
    for i in range(n):
        pair = input(f"Point {i+1}: ").split()
        if len(pair) != 2:
            raise ValueError("You must enter exactly two numbers per point.")
        xi = float(pair[0])
        yi = float(pair[1])
        x_vals.append(xi)
        y_vals.append(yi)

    # Ask user if they want to evaluate error at a point
    ans_eval = input("\nDo you want to evaluate the polynomial at a point? (y/n): ").strip().lower()
    if ans_eval == "y":
        x_eval = float(input("Enter x_eval: "))
        y_eval = float(input("Enter true y(x_eval) (for error comparison): "))
    else:
        x_eval = None
        y_eval = None

    # Ask user if they want a plot
    ans_plot = input("\nDo you want a plot? (y/n): ").strip().lower()
    want_plot = (ans_plot == "y")

    return x_vals, y_vals, x_eval, y_eval, want_plot


if __name__ == "__main__":
    # 1. Read all user input
    x_vals, y_vals, x_eval, y_eval, want_plot = read_user_input()

    # 2. Run Vandermonde interpolation
    result = vandermonde_interpolation(
        x_vals=x_vals,
        y_vals=y_vals,
        x_eval=x_eval,
        y_eval=y_eval
    )

    # 3. Print formatted report
    print_vandermonde_report(x_vals, y_vals, x_eval, y_eval, result)

    # 4. Optional plot
    if want_plot:
        maybe_plot(x_vals, y_vals, result["coeffs"], x_eval, y_eval)
