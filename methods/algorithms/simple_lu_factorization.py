import numpy as np

def eliminacion_gaussiana_simple_con_lu(A, b):
    """
    Implementa la eliminación gaussiana simple construyendo L y U "a la vista",
    mostrando las etapas, y trabajando con la matriz aumentada [A | b].
    No hace pivoteo.
    
    A: matriz cuadrada (n x n)
    b: vector (n,)
    Devuelve: matriz aumentada Ab al final del proceso (equivalente a [U | z])
    """
    # Convertimos a NumPy para trabajar más fácil
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    # 1. Crear matriz aumentada Ab = [A | b]
    Ab = np.zeros((n, n + 1), dtype=float)
    Ab[:, :n] = A
    Ab[:, n] = b

    # 2. Inicializar L y U
    L = np.identity(n, dtype=float)   # L con 1s en la diagonal
    U = np.zeros((n, n), dtype=float) # U vacía

    # --- IMPRESIÓN ETAPA 0 ---
    print("Etapa 0")
    print("Ab =")
    print(Ab)
    print("L =")
    print(L)
    print("U =")
    print(U)
    print("-" * 40)

    # Copiar la primera fila de A a U (equivalente a lo que dice tu pseudocódigo)
    U[0, :] = A[0, :]

    # 3. Etapas de eliminación
    for k in range(0, n - 1):  # k es la columna pivote
        # Comprobar pivote
        if Ab[k, k] == 0:
            raise ValueError(f"Pivote cero encontrado en la fila {k}. Se requiere pivoteo.")

        # Filas por debajo del pivote
        for i in range(k + 1, n):
            # multiplicador
            M_ik = Ab[i, k] / Ab[k, k]

            # guardar en L
            L[i, k] = M_ik

            # actualizar la fila i completa desde la columna k hasta el final (incluye b)
            Ab[i, k:] = Ab[i, k:] - M_ik * Ab[k, k:]

        # al final de la etapa, la fila k de Ab (solo la parte de A) ya es definitiva en U
        U[k, k:] = Ab[k, k:n]

        # --- IMPRESIÓN ETAPA k+1 (porque en tu pseudocódigo empieza en 1) ---
        print(f"Etapa {k + 1}")
        print("Ab =")
        print(Ab)
        print("L =")
        print(L)
        print("U =")
        print(U)
        print("-" * 40)

    # Asegurar que la última posición de U quede
    U[n - 1, n - 1] = Ab[n - 1, n - 1]

    print("--- ELIMINACIÓN COMPLETA ---")
    print("Ab final =")
    print(Ab)
    print("L final =")
    print(L)
    print("U final =")
    print(U)

    # Devolvemos la aumentada, como dice tu pseudocódigo
    return Ab, L, U

A = [[4, -1, 0, 3],
    [1,15.5,3,8],
    [0,-1.3,-4,1.1],
    [14,5,-2,30]]
b = [1, 1, 1, 1]

Ab, L, U = eliminacion_gaussiana_simple_con_lu(A, b)
        


