from sympy import symbols, sympify, lambdify

def falsePosition(a,b,f,tol,max_ite):
    x = symbols('x')
    allowed_locals = {"x":x}
    expr = sympify(f, locals=allowed_locals)
    f_num = lambdify(x, expr, modules="numpy")

    cont = 1
    ca = b-((f_num(b)*(b-a))/(f_num(b)-f_num(a)))

    print("|Iteration|  xi  |  f(xi) |  E  |")
    print("| ",1," | ",ca," | ",f_num(ca)," |   |",)

    while True:
        cont = cont +1
        c = ca
        if f_num(a)*f_num(b)<0:
            b=c
        else:
            a=c
        c = b-((f_num(b)*(b-a))/(f_num(b)-f_num(a)))
        eabs = abs(c-ca)
        print("| ",cont," | ",ca," | ",f_num(ca)," | ",eabs," |")
        if abs(c-ca)<tol:
            print("Root: ",c)
            return 1
        else:
            ca=c
        if cont == max_ite:
            print("Limit reached")
            return 0
        
reg_f = falsePosition(1,2,"x**3-x-2",10**-7,100)