import numpy as np

def imprimir_polinomio(a, n):
    polinomio = ""
    for i, coef in enumerate(a):
        exp = n - 1 - i # Exponente del término. Es en orden decreciente porque el vector a tiene los coeficientes en orden descendente (desde an-1 hasta a0)
        coef = round(coef, 4) # Redondear a 4 cifras para que no imprima términos muy largos (se puede quitar para que imprima el polinomio exacto)
        if coef == 0:
            continue # Se omiten términos nulos
        signo = " + " if coef > 0 and i > 0 else " " # Determina el signo del coeficiente
        if exp == 0:
            polinomio += f"{signo}{coef}"
        elif exp == 1:
            polinomio += f"{signo}{coef}
        else:
            polinomio += f"{signo}{coef}x^{exp}"
        
    return polinomio

def vandermonde(x, y, x_real=None, y_real=None):
    x = np.array(x) # Convierte la lista de puntos x en un arreglo de NumPy
    n = len(x) # Número de puntos
    A = [] # Matriz del sistema lineal
    b = np.array(y) # Vector de términos independientes

    # Se construye la matriz de Vandermonde
    for numero in x:
        fila = []
        # Cada fila contiene potencias decrecientes del valor de x
        for i in range(n - 1, -1, -1): # Esto recorre los números del n - 1 hasta 0, en orden descendente, que sería el exponente de cada número que va en la matriz de Vandermonde
            fila.append(numero**i)
        A.append(fila)

    A = np.array(A) # Convierte A a un arreglo NumPy
    a = np.linalg.solve(A, b) # Resuelve el sistema V*a = y (A*a = b) para obtener los coeficientes del polinomio
    print("La matriz de vandermonde es: ")
    print(A)

    print("Coeficientes: ")
    print(a)

    print("Polinomio:")
    pol = imprimir_polinomio(a, n)
    