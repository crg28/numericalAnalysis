from sympy import symbols, sympify, lambdify

def reglaFalsa(a,b,f,tol,max_ite):
    x = symbols('x')
    allowed_locals = {"x":x}
    expr = sympify(f, locals=allowed_locals)
    f_num = lambdify(x, expr, modules="numpy")

    cont = a
    ca = b-((f_num(b)*(b-a))/(f_num(b)-f_num(a)))

    while True:
        cont = cont +1
        c = ca
        if f_num(a)*f_num(b)<0:
            b=c
        else:
            a=c
        c = b-((f_num(b)*(b-a))/(f_num(b)-f_num(a)))
        if abs(c-ca)<tol:
            print("A: ",a," B: ",b," C: ",c)
            return 1
        else:
            ca=c
        if cont == max_ite:
            print("Limite de iteraciones alcanzado")
            return 0
        
reg_f = reglaFalsa(1,2,"x**3-x-2",10**-7,100)