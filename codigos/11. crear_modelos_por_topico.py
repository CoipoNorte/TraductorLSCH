import json
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parámetros
DIR_DATOS = 'datos/'  # Directorio con datos procesados y aumentados
MODELOS_PATH = 'modelos/topicos/'
HISTORIALES_PATH = 'historiales_topicos/'  # Nueva carpeta para historiales
TAMANO_OBJETIVO = (128, 128)  # Tamaño de la imagen de entrada
TIME_STEPS = 30  # Número de frames por secuencia
EPOCHS = 10
TAMANO_LOTE = 4

def construir_modelo(tamano_entrada=(TIME_STEPS, 128, 128, 3), num_clases=10):
    modelo = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=tamano_entrada),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(64, return_sequences=False),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_clases, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo

# Generador de secuencias de imágenes
def generar_secuencias(generator, time_steps):
    while True:
        x_batch = []
        y_batch = []
        
        for _ in range(generator.batch_size):
            x_sequence = []
            for _ in range(time_steps):
                x, y = generator.next()
                x_sequence.append(x[0])
            x_batch.append(np.array(x_sequence))
            y_batch.append(y[0])

        yield np.array(x_batch), np.array(y_batch)

# Entrenar el modelo
def entrenar_modelo_por_topico(topico, palabras):
    # Crear directorio para guardar el modelo del tópico y el historial
    os.makedirs(MODELOS_PATH, exist_ok=True)
    os.makedirs(HISTORIALES_PATH, exist_ok=True)
    
    modelo_path = os.path.join(MODELOS_PATH, f'modelo_{topico}.h5')
    historial_path = os.path.join(HISTORIALES_PATH, f'historial_{topico}.json')
    
    # Generador de datos para secuencias
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
    generador_entrenamiento = datagen.flow_from_directory(
        DIR_DATOS,
        target_size=TAMANO_OBJETIVO,
        classes=palabras,
        batch_size=1,
        class_mode='categorical',
        subset='training'
    )
    generador_validacion = datagen.flow_from_directory(
        DIR_DATOS,
        target_size=TAMANO_OBJETIVO,
        classes=palabras,
        batch_size=1,
        class_mode='categorical',
        subset='validation'
    )

    num_clases = generador_entrenamiento.num_classes
    modelo = construir_modelo(num_clases=num_clases)

    # Generadores de secuencias para entrenamiento y validación
    generador_secuencias_entrenamiento = generar_secuencias(generador_entrenamiento, TIME_STEPS)
    generador_secuencias_validacion = generar_secuencias(generador_validacion, TIME_STEPS)
    
    # Guardar el mejor modelo basado en la precisión de la validación
    checkpoint = ModelCheckpoint(modelo_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    # Entrenamiento del modelo
    historial = modelo.fit(
        generador_secuencias_entrenamiento,
        validation_data=generador_secuencias_validacion,
        epochs=EPOCHS,
        steps_per_epoch=len(generador_entrenamiento) // (TAMANO_LOTE * TIME_STEPS),
        validation_steps=len(generador_validacion) // (TAMANO_LOTE * TIME_STEPS),
        callbacks=[checkpoint]
    )

    # Guardar el historial de entrenamiento
    with open(historial_path, 'w') as f:
        json.dump(historial.history, f)

# Entrenar modelos para cada tópico basado en un JSON con los tópicos
def entrenar_modelos_por_topico():
    with open('topicos.json', 'r') as f:
        topicos = json.load(f)

    for topico, palabras in topicos.items():
        print(f"Entrenando modelo para el tópico: {topico}")
        entrenar_modelo_por_topico(topico, palabras)

if __name__ == "__main__":
    entrenar_modelos_por_topico()
