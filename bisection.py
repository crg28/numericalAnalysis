from sympy import symbols, sympify, lambdify
# Biseccion-Code

def bisection(a, b, f, tol, max_ite):
    cont = 0
    ca = (a+b)/2

    x = symbols('x')
    allowed_locals = {"x":x}
    expr = sympify(f, locals=allowed_locals)
    f_num = lambdify(x, expr, modules="numpy")
    
    print("|Iteration|  xi  |  f(xi) |  E  |")
    print("| ",0," | ",ca," | ",f_num(ca)," |   |",)

    while True:
        cont = cont + 1
        if f_num(a)*f_num(ca)<0:
            a=a
            b=ca
        else:
            a=ca
            b=b
        c = (b+a)/2
        eabs = abs(c-ca)
        print("| ",cont," | ",ca," | ",f_num(ca)," | ",eabs," |")
        if eabs<tol:
            print("A: ",a," B: ",b," C: ",c)
            return c
        else:
            ca=c
        if cont == max_ite:
            return "Limit reached"

bisec = bisection(-0.5,1,"x**2-x+1-E**x+99/200",0.000001,100)
print(bisec)