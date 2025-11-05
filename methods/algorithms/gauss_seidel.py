import numpy as np

def radio_espectral(T):
    """
    Calcula el radio espectral de una matriz T (el máximo de los valores absolutos
    de sus eigenvalores).
    """
    try:
        eigenvalores = np.linalg.eigvals(T)
        return np.max(np.abs(eigenvalores))
    except np.linalg.LinAlgError:
        return np.nan # En caso de error de convergencia

def gauss_seidel_matricial(A, b, x0, tolerancia, niter):
    
    # 6 decimales
    np.set_printoptions(precision=6, suppress=True, floatmode='fixed')
    
    iteracion = 0
    error = 1 # Error inicial
    
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).T
    x0 = np.array(x0, dtype=float).T
    
    # Descomposición y cálculo de T y C
    
    # Descomposición de la matriz A en D, L y U
    D = np.diag(np.diagonal(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    
    # Inversa de (D - L)
    inv_DL = np.linalg.inv(D - L)

    # Matriz de transición de Gauss-Seidel
    T = inv_DL @ U
    
    # Vector C de Gauss-Seidel
    C = inv_DL @ b
    
    # Radio espectral
    ro = radio_espectral(T)
    
    print("Gauss-Seidel\n")
    print("Resultados:\n")
    print("T:")
    # Imprimir la matriz T
    print(T) 
    
    print("\nC:")
    # Imprimir el vector C
    print(C) 
    
    print(f"\nradio espectral:\n{ro:.6f}\n")

    # Impresión de la tabla de iteraciones
    
    print("| iter |     E      | ")
    print("-" * (50)) # Separador

    x_str = "  ".join([f"{val:.6f}" for val in x0])
    print(f"|  {iteracion: >3} | {'-': >10} | {x_str}")

    while error > tolerancia and iteracion < niter:
        
        x1 = T @ x0 + C # Fórmula de Gauss-Seidel
        
        error = np.linalg.norm(x1 - x0, np.inf)
        
        x0 = x1 # Actualizamos para la siguiente iteración
        iteracion += 1

        # Imprimir la fila de la iteración actual
        x_str = "  ".join([f"{val:.6f}" for val in x0])
        # Formateamos el error a 1 decimal en notación científica (ej. 3.8e-01)
        print(f"| {iteracion: >4} | {error: >10.1e} | {x_str}")

    # Impresión de resultados finales
    
    print("-" * (50))
    if error < tolerancia:
        print(f"\nGauss-Seidel matricial: Solución encontrada en {iteracion} iteraciones.")
        print(f"Solución: {x0}")
    else:
        print(f"\nGauss-Seidel matricial: Fracasó en {niter} iteraciones.")
        
    return x0, ro

A_input = np.array([
    [4, -1, 0, 3],
    [1, 15.5, 3, 8],
    [0, -1.3, -4, 1.1],
    [14, 5, -2, 30]
])
    
b_input = np.array([1, 1, 1, 1])
 
x0_input = np.array([0, 0, 0, 0])

gauss_seidel_matricial(A_input, b_input, x0_input, tolerancia=1e-7, niter=40)