import math

def fixed_point(f, g, x0, tol=1e-7, max_iter=30):
    # Print table header
    print("{:<6} {:<15} {:<15} {:<15} {:<15}".format("iter", "xi", "g(xi)", "f(xi)", "E"))
    print("-" * 70)

    xn = x0
    for i in range(max_iter+1):
        gx = g(xn)
        fx = f(xn)

        if i == 0:
            # Print first iteration without error
            print("{:<6} {:<15.8f} {:<15.8f} {:<15.2e} {:<15}".format(i, xn, gx, fx, ""))
        else:
            # Compute error
            error = abs(gx - xn)
            # Print iteration values
            print("{:<6} {:<15.8f} {:<15.8f} {:<15.2e} {:<15.2e}".format(i, xn, gx, fx, error))

            # Check stopping criterion
            if error < tol:
                print(f"\nAn approximation of the root was found at {xn:.15f}")
                return xn, i

        # Update for next iteration
        xn = gx  

    # Print message if max iterations reached
    print("\nTolerance was not reached within the maximum number of iterations")
    return xn, max_iter


# Define functions
f = lambda x: math.log(math.sin(x)**2 + 1) - 0.5 - x
g = lambda x: math.log(math.sin(x)**2 + 1) - 0.5

# Initial value
x0 = -0.5

# Run method
fixed_point(f, g, x0, tol=1e-7, max_iter=30)