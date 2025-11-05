import numpy as np

def spline(x, y, d, x_real=None, y_real=None):
    n = len(x) # Cantidad de puntos
    # Matriz banda del método de spline
    A = np.zeros(((d+1)*(n-1), (d+1)*(n-1)))
    b = np.zeros((d+1)*(n-1)) # Vector de resultados

    cua = np.array(x)**2 # Eleva todas las x al cuadrado
    cub = np.array(x)**3 # Eleva todas las x al cubo

    c = 0 # Índice para las columnas de la matriz A
    h = 0 # Índice para las filas de la matriz A y el vector b
    
    if d == 1: # Spline lineal
        # Primera condición: que el polinomio en x_i dé y_i
        for i in range(n-1):
            A[h, c] = x[i] # coeficiente de a_i
            A[h, c+1] = 1 # coeficiente de b_i
            b[h] = y[i] # b_i
            c += 2 # Se avanzan dos columnas
            h += 1 # Se avanza una fila

        c = 0
        # Se repite pero empezando en p2
        for i in range(1, n):
            A[h, c] = x[i]
            A[h, c+1] = 1
            b[h] = y[i]
            c += 2
            h += 1

    elif d == 2: # Spline cuadrático
        for i in range(n-1):
            A[h, c] = cua[i]
            A[h, c+1] = x[i]
            A[h, c+2] = 1
            b[h] = y[i]
            c += 3
            h += 1

        c = 0
        for i in range(1, n):
            A[h, c] = cua[i]
            A[h, c+1] = x[i]
            A[h, c+2] = 1
            b[h] = y[i]
            c += 3
            h += 1

        c = 0
        for i in range(1, n-1):
            A[h, c] = 2*x[i]
            A[h, c+1] = 1
            A[h, c+3] = -2*x[i]
            A[h, c+4] = -1
            b[h] = 0
            c += 3
            h += 1

        A[h, 0] = 2
        b[h] = 0

    elif d == 3: # Spline cúbico
        # Primera condición: que el polinomio en x_i dé y_i
        for i in range(n-1):
            A[h, c] = cub[i] # coeficiente de a_i
            A[h, c+1] = cua[i] # coeficiente de b_i
            A[h, c+2] = x[i] # coeficiente de c_i
            A[h, c+3] = 1 # d_i
            b[h] = y[i]
            c += 4 # Se avanzan 4 columnas
            h += 1 # Se avanza una fila

        c = 0
        # Se repite pero empezando en p2
        for i in range(1, n):
            A[h, c] = cub[i]
            A[h, c+1] = cua[i]
            A[h, c+2] = x[i]
            A[h, c+3] = 1
            b[h] = y[i]
            c += 4
            h += 1

        c = 0
        # Condiciones de la primera derivada
        for i in range(1, n-1):
            # Derivada primer polinomio
            A[h, c] = 3*cua[i]
            A[h, c+1] = 2*x[i]
            A[h, c+2] = 1
            # Derivada del segundo polinomio negativa
            A[h, c+4] = -3*cua[i]
            A[h, c+5] = -2*x[i]
            A[h, c+6] = -1
            b[h] = 0 # Se iguala a 0
            c += 4
            h += 1

        c = 0
        # Condiciones de la segunda derivada
        for i in range(1, n-1):
            # Segunda derivada del primer polinomio
            A[h, c] = 6*x[i]
            A[h, c+1] = 2
            # Segunda derivada del segundo polinomio negativa
            A[h, c+4] = -6*x[i]
            A[h, c+5] = -2
            b[h] = 0 # Se iguala a 0
            c += 4
            h += 1

        # Segunda derivada en los extremos = 0
        # Segunda derivada del primer polinomio
        A[h, 0] = 6*x[0]
        A[h, 1] = 2
        b[h] = 0 # Se iguala a 0

        h += 1 # Se avanza a la siguiente fila
        
        # Segunda derivada del último polinomio
        A[h, c] = 6*x[-1]
        A[h, c+1] = 2
        b[h] = 0 # Se iguala a 0


    val = np.linalg.solve(A, b) # Se resuelve el sistema matricial
    # Se reorganizan los coeficientes en una tabla: una fila por tramo
    tabla = val.reshape((n-1, d+1)) 

    # Retornar calculos
    return tabla


# --- función para imprimir los trazadores
def imprimir_trazadores(tabla, d):
    """
    Imprime los polinomios de spline en el formato deseado.
    """
    # Para cada fila (tramo) en la tabla de coeficientes
    for i in range(tabla.shape[0]):
        coef = tabla[i] # Coeficientes del tramo actual
        polinomio = ""

        # Recorrer cada coeficiente (de j=0 a d)
        for j in range(d + 1):
            c = coef[j] # Coeficiente actual
            exp = d - j # Exponente correspondiente 
            
            # Omitir términos que son cero (excepto el término constante)
            if abs(c) < 1e-12 and exp != 0:
                continue

            # Formatear el coeficiente con su signo
            # El primer término no lleva '+' si es positivo
            if j == 0 or (len(polinomio) == 0 and c >= 0):
                term_str = f"{c:.6f}"
            else:
                # Añadir el signo + o - explícitamente
                term_str = f"+{c:.6f}" if c >= 0 else f"{c:.6f}"
            
            # Añadir variable y exponente
            if exp > 0:
                term_str += "x"
            if exp > 1:
                term_str += f"^{exp}"
            
            polinomio += term_str

        print(polinomio) # Imprimir el polinomio completo para este tramo


x = [-1, 0, 3, 4]
y = [15.5, 3, 8, 1]

# 6 Decimales
np.set_printoptions(precision=6, suppress=True, floatmode='fixed')

# --- Trazadores Lineales
print("Trazadores lineales\n")
print("Coeficientes de los trazadores:\n")
tabla_lin = spline(x, y, 1)
print(tabla_lin)
print("\nTrazadores:\n")
imprimir_trazadores(tabla_lin, 1)

# --- Trazadores Cuadráticos
print("\n\nTrazadores cuadraticos\n")
print("Coeficientes de los trazadores:\n")
tabla_cua = spline(x, y, 2)
print(tabla_cua)
print("\nTrazadores:\n")
imprimir_trazadores(tabla_cua, 2)

# --- Trazadores Cúbicos
print("\n\nTrazadores cúbicos\n")
print("Coeficientes de los trazadores:\n")
tabla_cub = spline(x, y, 3)
print(tabla_cub)
print("\nTrazadores:\n")
imprimir_trazadores(tabla_cub, 3)