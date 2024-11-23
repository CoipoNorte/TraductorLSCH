import os
from PIL import Image

# Configuración
DIR_SALIDA = 'datos/'
EXPECTED_SIZE = (256, 256)  # Tamaño esperado de las imágenes

def revisar_imagenes(dir_salida, expected_size=EXPECTED_SIZE):
    problemas = []
    for clase in os.listdir(dir_salida):
        ruta_clase = os.path.join(dir_salida, clase)
        if not os.path.isdir(ruta_clase):
            continue
        
        print(f"Revisando clase: {clase}")
        contador_imagenes = 0
        
        for sample_folder in os.listdir(ruta_clase):
            ruta_sample = os.path.join(ruta_clase, sample_folder)
            if not os.path.isdir(ruta_sample):
                continue
            
            for img_name in os.listdir(ruta_sample):
                img_path = os.path.join(ruta_sample, img_name)
                
                try:
                    img = Image.open(img_path)
                    if img.size != expected_size:
                        problemas.append((img_path, img.size))
                    contador_imagenes += 1
                except Exception as e:
                    print(f"Error al abrir {img_path}: {e}")
        
        print(f"Total imágenes en {clase}: {contador_imagenes}")

    if problemas:
        print("\nProblemas encontrados:")
        for img_path, size in problemas:
            print(f" - {img_path} tiene tamaño {size} en lugar de {expected_size}")
    else:
        print("\nTodas las imágenes tienen el tamaño esperado.")

if __name__ == "__main__":
    revisar_imagenes(DIR_SALIDA)
