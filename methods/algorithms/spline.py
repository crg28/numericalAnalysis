import numpy as np

def spline(x, y, d, x_real=None, y_real=None):
    n = len(x)  # Number of points
    # Band matrix for the spline method
    A = np.zeros(((d + 1) * (n - 1), (d + 1) * (n - 1)))
    b = np.zeros((d + 1) * (n - 1))  # Result vector

    cua = np.array(x) ** 2  # Square of x values
    cub = np.array(x) ** 3  # Cube of x values

    c = 0  # Column index for A
    h = 0  # Row index for A and b

    if d == 1:  # Linear spline
        # First condition: the polynomial at x_i must give y_i
        for i in range(n - 1):
            A[h, c] = x[i]      # Coefficient of a_i
            A[h, c + 1] = 1     # Coefficient of b_i
            b[h] = y[i]
            c += 2              # Move two columns forward
            h += 1              # Move one row down

        c = 0
        # Repeat starting from p2
        for i in range(1, n):
            A[h, c] = x[i]
            A[h, c + 1] = 1
            b[h] = y[i]
            c += 2
            h += 1

    elif d == 2:  # Quadratic spline
        for i in range(n - 1):
            A[h, c] = cua[i]
            A[h, c + 1] = x[i]
            A[h, c + 2] = 1
            b[h] = y[i]
            c += 3
            h += 1

        c = 0
        for i in range(1, n):
            A[h, c] = cua[i]
            A[h, c + 1] = x[i]
            A[h, c + 2] = 1
            b[h] = y[i]
            c += 3
            h += 1

        c = 0
        for i in range(1, n - 1):
            A[h, c] = 2 * x[i]
            A[h, c + 1] = 1
            A[h, c + 3] = -2 * x[i]
            A[h, c + 4] = -1
            b[h] = 0
            c += 3
            h += 1

        A[h, 0] = 2
        b[h] = 0

    elif d == 3:  # Cubic spline
        # First condition: the polynomial at x_i must give y_i
        for i in range(n - 1):
            A[h, c] = cub[i]    # Coefficient of a_i
            A[h, c + 1] = cua[i]  # Coefficient of b_i
            A[h, c + 2] = x[i]  # Coefficient of c_i
            A[h, c + 3] = 1     # d_i
            b[h] = y[i]
            c += 4              # Move 4 columns forward
            h += 1

        c = 0
        # Repeat starting from p2
        for i in range(1, n):
            A[h, c] = cub[i]
            A[h, c + 1] = cua[i]
            A[h, c + 2] = x[i]
            A[h, c + 3] = 1
            b[h] = y[i]
            c += 4
            h += 1

        c = 0
        # First derivative conditions
        for i in range(1, n - 1):
            # First derivative of the first polynomial
            A[h, c] = 3 * cua[i]
            A[h, c + 1] = 2 * x[i]
            A[h, c + 2] = 1
            # Negative derivative of the second polynomial
            A[h, c + 4] = -3 * cua[i]
            A[h, c + 5] = -2 * x[i]
            A[h, c + 6] = -1
            b[h] = 0
            c += 4
            h += 1

        c = 0
        # Second derivative conditions
        for i in range(1, n - 1):
            # Second derivative of the first polynomial
            A[h, c] = 6 * x[i]
            A[h, c + 1] = 2
            # Negative second derivative of the second polynomial
            A[h, c + 4] = -6 * x[i]
            A[h, c + 5] = -2
            b[h] = 0
            c += 4
            h += 1

        # Second derivative at endpoints = 0
        # Second derivative of the first polynomial
        A[h, 0] = 6 * x[0]
        A[h, 1] = 2
        b[h] = 0

        h += 1  # Move to next row

        # Second derivative of the last polynomial
        A[h, c] = 6 * x[-1]
        A[h, c + 1] = 2
        b[h] = 0

    val = np.linalg.solve(A, b)  # Solve the matrix system
    # Reshape coefficients into a table: one row per segment
    table = val.reshape((n - 1, d + 1))

    # Return results
    return table


# --- Function to print spline equations
def print_splines(table, d):
    """
    Prints the spline polynomials in the desired format.
    """
    for i in range(table.shape[0]):
        coef = table[i]
        polynomial = ""

        for j in range(d + 1):
            c = coef[j]
            exp = d - j

            # Skip zero terms (except constant)
            if abs(c) < 1e-12 and exp != 0:
                continue

            # Format coefficient sign
            if j == 0 or (len(polynomial) == 0 and c >= 0):
                term_str = f"{c:.6f}"
            else:
                term_str = f"+{c:.6f}" if c >= 0 else f"{c:.6f}"

            # Add variable and exponent
            if exp > 0:
                term_str += "x"
            if exp > 1:
                term_str += f"^{exp}"

            polynomial += term_str

        print(polynomial)


x = [-1, 0, 3, 4]
y = [15.5, 3, 8, 1]

np.set_printoptions(precision=6, suppress=True, floatmode='fixed')

# --- Linear Splines
print("Linear Splines\n")
print("Spline Coefficients:\n")
table_lin = spline(x, y, 1)
print(table_lin)
print("\nSplines:\n")
print_splines(table_lin, 1)

# --- Quadratic Splines
print("\n\nQuadratic Splines\n")
print("Spline Coefficients:\n")
table_quad = spline(x, y, 2)
print(table_quad)
print("\nSplines:\n")
print_splines(table_quad, 2)

# --- Cubic Splines
print("\n\nCubic Splines\n")
print("Spline Coefficients:\n")
table_cub = spline(x, y, 3)
print(table_cub)
print("\nSplines:\n")
print_splines(table_cub, 3)
