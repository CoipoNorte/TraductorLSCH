import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Parámetros
DIR_DATOS = 'datos_aumentados/'  # Directorio con datos procesados y aumentados
NOMBRE_MODELO = 'modelos/modeloLSCH_general.h5'
HISTORIAL_PATH = 'modelos/historial_entrenamiento.json'
TAMANO_OBJETIVO = (256, 256)  # Tamaño de imagen ajustado a la resolución 256x256
EPOCHS = 10
TAMANO_LOTE = 2  # Lotes de secuencias, reducido para CPU
NUM_FRAMES = 30  # Número de frames por secuencia

# Construcción del modelo CNN + LSTM
def construir_modelo(tamano_entrada=(NUM_FRAMES, 256, 256, 3), num_clases=10):
    modelo = Sequential()
    
    # TimeDistributed para aplicar la CNN en cada frame de la secuencia
    modelo.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=tamano_entrada))
    modelo.add(TimeDistributed(MaxPooling2D((2, 2))))
    modelo.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    modelo.add(TimeDistributed(MaxPooling2D((2, 2))))
    modelo.add(TimeDistributed(Flatten()))
    
    # LSTM para capturar relaciones temporales
    modelo.add(LSTM(128, activation='relu', return_sequences=False))

    # Capa densa para clasificación final
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(num_clases, activation='softmax'))
    
    # Compilar el modelo
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo

# Generar secuencias de datos en lotes pequeños
def generar_secuencias(generator, num_frames=NUM_FRAMES, batch_size=TAMANO_LOTE):
    while True:
        X, y = [], []
        for _ in range(batch_size):
            frames, etiqueta = [], None
            for _ in range(num_frames):
                batch = next(generator)
                frames.append(batch[0][0])  # Frame individual
                etiqueta = batch[1][0]      # Etiqueta para la secuencia
            X.append(np.array(frames))
            y.append(etiqueta)
        yield np.array(X), np.array(y)

# Función para entrenar el modelo
def entrenar_modelo():
    # Configuración de Data Augmentation para escalado
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2
    )

    # Generadores de entrenamiento y validación
    generador_entrenamiento = datagen.flow_from_directory(
        DIR_DATOS,
        target_size=TAMANO_OBJETIVO,
        batch_size=1,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    generador_validacion = datagen.flow_from_directory(
        DIR_DATOS,
        target_size=TAMANO_OBJETIVO,
        batch_size=1,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    # Número de clases
    num_clases = generador_entrenamiento.num_classes
    modelo = construir_modelo(tamano_entrada=(NUM_FRAMES, 256, 256, 3), num_clases=num_clases)

    # Guardar el mejor modelo según la precisión en validación
    checkpoint = ModelCheckpoint(NOMBRE_MODELO, monitor='val_accuracy', save_best_only=True, verbose=1)

    # Generadores de secuencias de entrenamiento y validación
    secuencias_entrenamiento = generar_secuencias(generador_entrenamiento, NUM_FRAMES, batch_size=TAMANO_LOTE)
    secuencias_validacion = generar_secuencias(generador_validacion, NUM_FRAMES, batch_size=TAMANO_LOTE)

    # Entrenamiento del modelo
    historial = modelo.fit(
        secuencias_entrenamiento,
        steps_per_epoch=generador_entrenamiento.samples // (NUM_FRAMES * TAMANO_LOTE),
        validation_data=secuencias_validacion,
        validation_steps=generador_validacion.samples // (NUM_FRAMES * TAMANO_LOTE),
        epochs=EPOCHS,
        callbacks=[checkpoint],
        workers=1,               # Establecemos a 1 worker
        use_multiprocessing=False # Evitamos el multiprocesamiento
    )

    # Guardar el historial de entrenamiento
    with open(HISTORIAL_PATH, 'w') as f:
        json.dump(historial.history, f)

if __name__ == "__main__":
    entrenar_modelo()
