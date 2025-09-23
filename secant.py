import math

def secant(f, x0, x1, tol=1e-7, max_iter=100):
    """
    Secant method to solve f(x) = 0.
    Prints a table with iteration, xi, f(xi), and absolute error (E).
    The stopping condition is based only on the absolute error.
    """
    print("\nSecant\n")
    print("Result table:\n")
    print("| iter |          xi  |        f(xi) |          E |")
    
    for i in range(max_iter + 1):
        # Iteration 0: initial x0
        if i == 0:
            f_x0 = f(x0)
            print(f"| {i:3d}  | {x0:11.9f} | {f_x0:11.1e} |            |")
        
        # Iteration 1: initial x1
        elif i == 1:
            f_x1 = f(x1)
            print(f"| {i:3d}  | {x1:11.9f} | {f_x1:11.1e} |            |")
        
        # From iteration 2 onward
        else:
            f_x0, f_x1 = f(x0), f(x1)
            if f_x1 - f_x0 == 0:
                raise ZeroDivisionError("Division by zero in the secant formula.")
            
            xi = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
            f_xi = f(xi)
            e = abs(xi - x1)

            print(f"| {i:3d}  | {xi:11.9f} | {f_xi:11.1e} | {e:10.1e} |")

            if e < tol:
                print(f"\nAn approximation of the root was found at {xi}")
                return xi, i

            # Update values
            x0, x1 = x1, xi

    print("\nDid not converge within the maximum number of iterations")
    return x1, max_iter


# Example usage
f = lambda x: math.log(math.sin(x)**2 + 1) - 0.5

x0, x1 = 0.5, 1.0
root, iterations = secant(f, x0, x1, tol=1e-7)