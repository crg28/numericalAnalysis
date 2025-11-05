import numpy as np

def newtonint(x, y, x_real=None, y_real=None):
    n = len(x) # Cantidad de puntos
    
    # imprimir con 6 decimales fijos
    np.set_printoptions(precision=6, suppress=True, floatmode='fixed')
    
    # Convertir x e y a arrays de NumPy
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Crear una tabla de ceros de tamaño n x (n+1)
    tabla = np.zeros((n, n+1))

    for i in range(n):
        tabla[i, 0] = x[i] # Primera columna de la tabla son los valores de x
        tabla[i, 1] = y[i] # Segunda columna de la tabla son los valores de y

    # Calcular las diferencias divididas
    for j in range(2, n + 1): 
        for i in range(j-1, n): 
            tabla[i, j] = (tabla[i, j-1] - tabla[i-1, j-1]) / (tabla[i, 0] - tabla[i-j+1, 0])

    # Extraer coeficientes como un array de NumPy
    coef = np.array([tabla[i, i + 1] for i in range(n)])

    # Extraer la tabla de diferencias (sin la columna de x)
    diferencias_div = np.delete(tabla, 0 , axis=1)

    # --- IMPRESIÓN FORMATEADA ---
    print("Tabla de diferencias divididas:")
    print(diferencias_div) 
    print("\nCoeficientes del polinomio de Newton:")
    print(coef)

    # Generar el string del Polinomio de Newton ---
    print("\nPolinomio de Newton:")
    
    # Empezar con el primer coeficiente (c0)
    pol_str = f"{coef[0]:.6f}"
    
    # Recorrer los demás coeficientes para añadir los términos
    for i in range(1, n):
        # Añadir el coeficiente (c_i) con su signo
        if coef[i] >= 0:
            pol_str += f"+{coef[i]:.6f}"
        else:
            pol_str += f"{coef[i]:.6f}" # El signo negativo ya viene
        
        # Añadir los términos (x - x_j)
        for j in range(i):
            if x[j] == 0:
                pol_str += "x" # Si x_j es 0, solo pone 'x'
            elif x[j] > 0:
                pol_str += f"(x-{x[j]})"
            else:
                pol_str += f"(x+{-x[j]})" # Si x_j es -1, pone (x+1)
    
    print(pol_str)
    
    return pol_str, coef

x = [-1, 0, 3, 4]
y = [15.5, 3, 8, 1]
newtonint(x, y)