from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from PIL import Image  # Usaremos PIL directamente para manipular imágenes
import os

# Configuración
DIR_ENTRADA = 'capturas/'
DIR_SALIDA = 'datos/'
TAMANO_OBJETIVO = (256, 256)  # Ajuste a una resolución mayor para mejorar la claridad

def preprocesar_y_copiar_imagenes(dir_entrada, dir_salida, tamano_objetivo=TAMANO_OBJETIVO):
    os.makedirs(dir_salida, exist_ok=True)
    
    for clase in os.listdir(dir_entrada):
        ruta_clase = os.path.join(dir_entrada, clase)
        if not os.path.isdir(ruta_clase):
            continue
        
        # Crear carpeta para la clase en la salida
        dir_clase_salida = os.path.join(dir_salida, clase)
        os.makedirs(dir_clase_salida, exist_ok=True)

        for sample_folder in os.listdir(ruta_clase):
            ruta_sample = os.path.join(ruta_clase, sample_folder)
            if not os.path.isdir(ruta_sample):
                continue
            
            # Crear carpeta para cada muestra de la clase
            dir_sample_salida = os.path.join(dir_clase_salida, sample_folder)
            os.makedirs(dir_sample_salida, exist_ok=True)
            
            # Procesar cada imagen en la muestra
            for img_name in os.listdir(ruta_sample):
                img_path = os.path.join(ruta_sample, img_name)
                
                # Cargar la imagen en su tamaño original
                img = load_img(img_path)
                
                # Mantener la relación de aspecto al redimensionar
                img = img_to_array(img)
                original_height, original_width = img.shape[:2]
                
                # Calcular la nueva escala manteniendo la relación de aspecto
                scale = min(tamano_objetivo[0] / original_width, tamano_objetivo[1] / original_height)
                new_size = (int(original_width * scale), int(original_height * scale))
                
                # Redimensionar y luego ajustar al tamaño deseado sin distorsionar
                img = Image.fromarray(img.astype('uint8')).resize(new_size)
                
                # Crear una imagen con fondo negro y pegar la imagen redimensionada en el centro
                final_img = Image.new("RGB", tamano_objetivo)
                final_img.paste(img, ((tamano_objetivo[0] - new_size[0]) // 2, (tamano_objetivo[1] - new_size[1]) // 2))
                
                # Guardar la imagen procesada en el directorio de salida
                save_path = os.path.join(dir_sample_salida, img_name)
                final_img.save(save_path)

if __name__ == "__main__":
    preprocesar_y_copiar_imagenes(DIR_ENTRADA, DIR_SALIDA)
