# Librerías necesarias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# FUNCIONES OBJETIVO - PROPORCIONADAS
# ================================================================

# --- Funcion 1D ---
def f_1d(x):
    """f(x) = (x - 3)^2 + 5"""
    return (x - 3)**2 + 5

def df_1d(x):
    """Derivada de f: f'(x) = 2(x - 3)"""
    return 2 * (x - 3)

# --- Funcion 2D ---
def f_2d(x, y):
    """f(x, y) = x^2 + y^2 - 4x - 2y + 5"""
    return x**2 + y**2 - 4*x - 2*y + 5

def grad_2d(x, y):
    """Gradiente de f: [2x - 4, 2y - 2]"""
    return np.array([2*x - 4, 2*y - 2])


def gradiente_descendente_1d(x_inicial, learning_rate, max_iter=1000, tolerancia=1e-6):
    """
    Implementa gradiente descendente para la funcion f(x) = (x - 3)^2 + 5
    
    Parametros:
    -----------
    x_inicial : float de manera aleatoria
        Punto de inicio
    learning_rate : float
        Tamano del paso (alpha)
    max_iter : int
        Numero maximo de iteraciones
    tolerancia : float
        Si |x_nuevo - x_actual| < tolerancia, se considera convergido
    
    Retorna:
    --------
    dict con:
        'x_final': float - Valor final de x
        'f_final': float - Valor de f(x_final)
        'iteraciones': int - Numero de iteraciones realizadas
        'convergido': bool - Si el algoritmo convergio
        'historial_x': list - Valores de x en cada iteracion
        'historial_f': list - Valores de f(x) en cada iteracion
    """

    x_actual = float(np.random.uniform(-10, 10)) if x_inicial is None else x_inicial
    historial_x = [x_actual]
    historial_f = [f_1d(x_actual)]
    convergido = False

    for i in range(max_iter):
        g = df_1d(x_actual)
        x_nuevo = x_actual - learning_rate * g

        historial_x.append(x_nuevo)
        historial_f.append(f_1d(x_nuevo))

        if abs(x_nuevo - x_actual) < tolerancia:
            convergido = True
            x_actual = x_nuevo
            iteraciones = i + 1
            break

        x_actual = x_nuevo
    else:
        iteraciones = max_iter

    return {
        'x_final': x_actual,
        'f_final': f_1d(x_actual),
        'iteraciones': iteraciones,
        'convergido': convergido,
        'historial_x': historial_x,
        'historial_f': historial_f
    }
    
def gradiente_descendente_2d(x_inicial, y_inicial, learning_rate, max_iter=1000, tolerancia=1e-6):
    """
    Implementa gradiente descendente para f(x,y) = x^2 + y^2 - 4x - 2y + 5
    
    Parametros:
    -----------
    x_inicial : float
        Valor inicial de x
    y_inicial : float
        Valor inicial de y
    learning_rate : float
        Tamano del paso (alpha)
    max_iter : int
        Numero maximo de iteraciones
    tolerancia : float
        Si la norma del gradiente < tolerancia, se considera convergido
    
    Retorna:
    --------
    dict con:
        'x_final': float
        'y_final': float
        'f_final': float - Valor de f(x_final, y_final)
        'iteraciones': int
        'convergido': bool
        'historial_x': list
        'historial_y': list
        'historial_f': list
    """
    
    x_actual, y_actual = x_inicial, y_inicial
    historial_x = [x_actual]
    historial_y = [y_actual]
    historial_f = [f_2d(x_actual, y_actual)]
    convergido = False

    for i in range(max_iter):
        g = grad_2d(x_actual, y_actual)
        x_nuevo = x_actual - learning_rate * g[0]
        y_nuevo = y_actual - learning_rate * g[1]

        historial_x.append(x_nuevo)
        historial_y.append(y_nuevo)
        historial_f.append(f_2d(x_nuevo, y_nuevo))

        if np.linalg.norm(g) < tolerancia:
            convergido = True
            x_actual, y_actual = x_nuevo, y_nuevo
            iteraciones = i + 1
            break

        x_actual, y_actual = x_nuevo, y_nuevo
    else:
        iteraciones = max_iter

    return {
        'x_final': x_actual,
        'y_final': y_actual,
        'f_final': f_2d(x_actual, y_actual),
        'iteraciones': iteraciones,
        'convergido': convergido,
        'historial_x': historial_x,
        'historial_y': historial_y,
        'historial_f': historial_f
    }


def generar_CSV():
    filas = []

    # Experimentos 1D
    for lr in [0.001, 0.01, 0.1, 0.5, 0.9, 1.0, 1.5]:
        res = gradiente_descendente_1d(x_inicial=-2.0, learning_rate=lr, max_iter=200)
        filas.append({
            'learning_rate': lr, 'dimension': '1D',
            'x_inicial': -2.0, 'y_inicial': np.nan,
            'x_final': res['x_final'], 'y_final': np.nan,
            'valor_minimo': res['f_final'],
            'iteraciones': res['iteraciones'],
            'convergido': res['convergido']
        })

    # Experimentos 2D
    for lr in [0.001, 0.01, 0.1, 0.5]:
        for (xi, yi) in [(-1.0, 4.0), (5.0, -1.0), (0.0, 0.0)]:
            res = gradiente_descendente_2d(xi, yi, learning_rate=lr, max_iter=500)
            filas.append({
                'learning_rate': lr, 'dimension': '2D',
                'x_inicial': xi, 'y_inicial': yi,
                'x_final': res['x_final'], 'y_final': res['y_final'],
                'valor_minimo': res['f_final'],
                'iteraciones': res['iteraciones'],
                'convergido': res['convergido']
            })

    df_experimentos = pd.DataFrame(filas)
    df_experimentos.to_csv('experimentos_gd.csv', index=False)
    print("Archivo experimentos_gd.csv generado correctamente")


def main():
    generar_CSV()


if __name__ == "__main__":
    main()

        