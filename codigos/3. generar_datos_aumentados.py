from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img, save_img
import os

# Configuración
DIR_ENTRADA = 'datos/'  # Carpeta de datos procesados originales
DIR_SALIDA = 'datos_aumentados/'  # Carpeta de salida para datos originales y aumentados
TAMANO_OBJETIVO = (256, 256)  # Resolución de imagen de 256x256
NUM_AUMENTOS_POR_IMAGEN = 5  # Número de imágenes aumentadas por imagen original

# Configuración de Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=5,  # Rotación mínima para evitar giros excesivos
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1]
)

# Crear el directorio de salida si no existe
os.makedirs(DIR_SALIDA, exist_ok=True)

# Iterar sobre cada clase en el directorio de entrada
for clase in os.listdir(DIR_ENTRADA):
    ruta_clase = os.path.join(DIR_ENTRADA, clase)
    if not os.path.isdir(ruta_clase):
        continue
    
    # Crear carpeta para la clase en la salida
    dir_clase_salida = os.path.join(DIR_SALIDA, clase)
    os.makedirs(dir_clase_salida, exist_ok=True)

    # Iterar sobre cada muestra (sample) en la clase
    for sample_folder in os.listdir(ruta_clase):
        ruta_sample = os.path.join(ruta_clase, sample_folder)
        if not os.path.isdir(ruta_sample):
            continue
        
        # Copiar el sample original a la carpeta de salida
        dir_sample_salida_original = os.path.join(dir_clase_salida, sample_folder)
        os.makedirs(dir_sample_salida_original, exist_ok=True)

        # Copiar todas las imágenes originales al nuevo directorio de salida
        for img_name in os.listdir(ruta_sample):
            img_path = os.path.join(ruta_sample, img_name)
            img = load_img(img_path, target_size=TAMANO_OBJETIVO)
            save_img(os.path.join(dir_sample_salida_original, img_name), img)

        # Crear una nueva carpeta para el sample aumentado
        dir_sample_salida_aumentado = os.path.join(dir_clase_salida, f"{sample_folder}_aumentado")
        os.makedirs(dir_sample_salida_aumentado, exist_ok=True)

        # Generar imágenes aumentadas para cada imagen en el sample original
        for img_name in os.listdir(ruta_sample):
            img_path = os.path.join(ruta_sample, img_name)
            img = load_img(img_path, target_size=TAMANO_OBJETIVO)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)  # Cambiar forma a (1, ancho, alto, canales)

            # Generar imágenes aumentadas y guardarlas en el nuevo sample
            contador = 0
            for batch in datagen.flow(x, batch_size=1):
                imagen_aumentada = array_to_img(batch[0])
                aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{contador}.jpg"
                save_img(os.path.join(dir_sample_salida_aumentado, aug_img_name), imagen_aumentada)
                contador += 1
                if contador >= NUM_AUMENTOS_POR_IMAGEN:
                    break

print("Generación de datos aumentados completa.")
