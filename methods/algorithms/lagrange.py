import numpy as np
# Asumimos que esta función existe y formatea un array de 
# coeficientes en un string legible, como en tu ejemplo.
from vandermonde import imprimir_polinomio

def lagrange(x, y, x_real=None, y_real=None):
    n = len(x) # Número de puntos

    # L_i(x) puros (sin y_i)
    L_tabla = np.zeros((n, n))
    
    # polinomio final simplificado
    pol_final_coeffs = np.zeros(n) 

    print("Polinomios interpolantes de Lagrange:")

    # Recorremos cada punto (x_i, y_i)
    for i in range(n):
        Li = np.array([1.0]) # Se inicializa el polinomio numerador
        denominador = 1.0 # Se inicializa el denominador

        # Loop para construir el numerador y denominador de L_i(x)
        for j in range(n):
            if j != i:
                paux = np.array([1.0, -x[j]])  # (x - x_j)
                Li = np.convolve(Li, paux) # Multiplicamos acumulativamente Li por (x - x_j)
                denominador *= (x[i] - x[j]) # Acumulamos el denominador

        # Guardar L_i(x) PURO
        # Calculamos los coeficientes de L_i(x) = Li / denominador
        Li_coeffs = Li / denominador
        
        # Guardamos los coeficientes de L_i(x) en nuestra tabla
        L_tabla[i, n - len(Li_coeffs):] = Li_coeffs

        # Usamos tu función para imprimir el polinomio L_i que acabamos de calcular
        print(f"L{i}: {imprimir_polinomio(L_tabla[i], n)}")

        # Acumular el polinomio final ---
        # Calculamos P(x) = P(x) + y_i * L_i(x)
        pol_final_coeffs = pol_final_coeffs + y[i] * L_tabla[i]

    print("\nPolinomio:")
    pol_str_parts = []
    for i in range(n):
        pol_str_parts.append(f"{y[i]}*L{i}")
    print(" + ".join(pol_str_parts))
    
    # polinomio simplificado
    print("\nPolinomio (simplificado):")
    polinomio_simplificado = imprimir_polinomio(pol_final_coeffs, n)
    print(polinomio_simplificado)
    
    return polinomio_simplificado, L_tabla

x = [-1, 0, 3, 4]
y = [15.5, 3, 8, 1]
lagrange(x, y)