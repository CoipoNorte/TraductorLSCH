import matplotlib.pyplot as plt
import json
import os
import numpy as np

HISTORIAL_PATH = 'modelos/historial_entrenamiento.json'

def suavizar_datos(datos, factor=0.8):
    """Aplicar promedio móvil para suavizar datos."""
    suavizado = []
    for i, valor in enumerate(datos):
        if i == 0:
            suavizado.append(valor)
        else:
            suavizado.append(suavizado[-1] * factor + valor * (1 - factor))
    return suavizado

def mostrar_datos_entrenamiento(historial_path=HISTORIAL_PATH, guardar=False):
    if not os.path.exists(historial_path):
        print(f"El archivo de historial {historial_path} no existe.")
        return
    
    with open(historial_path, 'r') as f:
        historial = json.load(f)
    
    # Suavizar datos para una mejor visualización
    precision_entrenamiento = suavizar_datos(historial['accuracy'])
    precision_validacion = suavizar_datos(historial['val_accuracy'])
    perdida_entrenamiento = suavizar_datos(historial['loss'])
    perdida_validacion = suavizar_datos(historial['val_loss'])
    
    plt.figure(figsize=(14, 6))

    # Gráfico de Precisión
    plt.subplot(1, 2, 1)
    plt.plot(precision_entrenamiento, label='Precisión Entrenamiento', linestyle='-', color='b')
    plt.plot(precision_validacion, label='Precisión Validación', linestyle='--', color='orange')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Precisión durante el Entrenamiento y Validación')
    plt.legend()

    # Gráfico de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(perdida_entrenamiento, label='Pérdida Entrenamiento', linestyle='-', color='b')
    plt.plot(perdida_validacion, label='Pérdida Validación', linestyle='--', color='orange')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Pérdida durante el Entrenamiento y Validación')
    plt.legend()

    # Mostrar o guardar la gráfica
    if guardar:
        plt.savefig('historial_entrenamiento.png')
    plt.show()

if __name__ == "__main__":
    mostrar_datos_entrenamiento()
