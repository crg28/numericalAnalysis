from sympy import symbols, sympify, lambdify
# Biseccion-Code
# Para implementar biseccion debo recibir a y b

def biseccion(a, b, f, tol, max_ite):
    cont = 0
    ca = (a+b)/2

    x = symbols('x')
    allowed_locals = {"x":x}
    expr = sympify(f, locals=allowed_locals)
    f_num = lambdify(x, expr, modules="numpy")
    
    while True:
        cont = cont + 1
        if f_num(a)*f_num(ca)<0:
            a=a
            b=ca
        else:
            a=ca
            b=b
        c = (b+a)/2
        if abs(c-ca)<tol:
            print("A: ",a," B: ",b," C: ",c)
            return c
        else:
            ca=c
        if cont == max_ite:
            return "Limite alcanzado"

bisec = biseccion(-0.5,1,"x**2-x+1-E**x+99/200",0.000001,100)
print(bisec)