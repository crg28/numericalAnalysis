from sympy import symbols, sympify, lambdify

def rootsMultiplicity(x0,f,f1,f2,tol,max_ite):
    cont = 0
    x = symbols('x')
    allowed_locals = {"x":x}
    expr = sympify(f, locals=allowed_locals)
    f_num = lambdify(x, expr, modules="numpy")
    expr = sympify(f1, locals=allowed_locals)
    f1_num = lambdify(x, expr, modules="numpy")
    expr = sympify(f2, locals=allowed_locals)
    f2_num = lambdify(x, expr, modules="numpy")

    print("|Iteration|  xi  |  f(xi) |  E  |")
    print("| ",0," | ",x0," | ",f_num(x0)," |   |",)

    while True:
        cont = cont + 1
        x1 = x0-((f_num(x0)*f1_num(x0)/(f1_num(x0)**2-(f_num(x0)*f2_num(x0)))))

        eabs = abs(x1-x0)
        print("| ",cont," | ",x0," | ",f_num(x0)," | ",eabs," |")
        if abs(x1-x0)<tol:
            print("Raíz: ",x1)
            return 1
        else:
            x0 = x1
        if cont == max_ite:
            print("Limite alcanzado")
            return 0
        
multRa = rootsMultiplicity(-2.45,"1-sin(x)**2+x+1.85","-sin(2*x)+1","-2*cos(2*x)",10**-7,100)