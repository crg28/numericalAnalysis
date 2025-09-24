from sympy import symbols, sympify, lambdify

def incremental_search(x0, deltaX, f, maximum_iterations):
    x = symbols('x')
    allowed_locals = {"x":x}
    expr = sympify(f, locals=allowed_locals)
    f_num = lambdify(x, expr, modules="numpy")
    
    counter = 0
    
    print("|Iteration|  xi  |  f(xi) |")
    print("| ",0," | ",x0," | ",f_num(x0),"|")
    while True:
        counter = counter + 1
        x1 = x0 +deltaX
        print("| ",counter," | ",x1," | ",f_num(x1),"|")
        if f_num(x1)*f_num(x0)<0:
            print("In [",x0,",",x1,"] there is at least one root")
            return [x0,x1]
        else:
            x0=x1
        if counter == maximum_iterations:
            print("Limit reached")
            return 0

multRa = incremental_search(-2.45, 0.5, "1-sin(x)**2+x+1.85", 100)