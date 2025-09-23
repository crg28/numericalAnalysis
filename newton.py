import math

# Define the function and its derivative
f = lambda x: math.log(math.sin(x)**2 + 1) - 0.5
df = lambda x: (2 * math.sin(x) * math.cos(x)) / (math.sin(x)**2 + 1)

# Newton method
def newton(x0, tol=1e-7, max_iter=100):
    print("Newton\n")
    print("Tabla de resultados:\n")
    print("| iter|          xi |       f(xi) |          E |")
    
    xi = x0
    for i in range(max_iter):
        fxi = f(xi)
        
        # First iteration (i=0) has no error
        if i == 0:
            print(f"| {i:3d} | {xi:11.9f} | {fxi:11.1e} |            |")
        else:
            E = abs(xi - x_prev)
            print(f"| {i:3d} | {xi:11.9f} | {fxi:11.1e} | {E:10.1e} |")
        
        if abs(fxi) < tol:
            break
        
        # Newton step
        x_prev = xi
        xi = xi - fxi/df(xi)
    
    print(f"\nSe encontró una aproximación de la raiz en {xi}")

# Run the method
newton(0.5)