import os
import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic
from text_to_speech import pronunciar_palabra

# Configuración
TAMANO_OBJETIVO = (128, 128)
TIME_STEPS = 30
THRESHOLD = 0.7
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_POS = (10, 30)
FONT_SIZE = 0.8
MODELOS_PATH = 'modelos/topicos/'
JSON_TOPICOS = 'topicos.json'

# Detección de manos con MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Verificar si hay una mano en pantalla
def there_hand(results):
    return results.left_hand_landmarks or results.right_hand_landmarks

# Cargar clases de cada tópico desde el JSON
def cargar_clases(topico):
    with open(JSON_TOPICOS, 'r') as f:
        topicos = json.load(f)
    return topicos.get(topico, [])

# Traducir en tiempo real con el modelo de CNN basado en secuencias
def traducir_en_tiempo_real(topico):
    cap = cv2.VideoCapture(0)
    kp_sequence = []

    # Cargar el modelo del tópico seleccionado
    model_path = os.path.join(MODELOS_PATH, f"modelo_{topico}.h5")
    modelo = load_model(model_path)

    # Cargar clases del tópico
    acciones = cargar_clases(topico)
    
    ultima_palabra = ""
    with Holistic() as holistic_model:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic_model)

            # Verificar si hay una mano en pantalla
            if there_hand(results):
                # Capturar y preprocesar el frame como imagen
                img = cv2.resize(frame, TAMANO_OBJETIVO)
                img = img / 255.0
                kp_sequence.append(img)

                # Si acumulamos la secuencia completa, hacemos la predicción
                if len(kp_sequence) == TIME_STEPS:
                    secuencia_array = np.expand_dims(np.array(kp_sequence), axis=0)
                    
                    # Predicción para la secuencia completa
                    prediccion = modelo.predict(secuencia_array)[0]
                    indice_predicho = np.argmax(prediccion)
                    confianza = prediccion[indice_predicho]

                    # Validar confianza y actualizar la traducción
                    if confianza > THRESHOLD:
                        palabra = acciones[indice_predicho]
                        if palabra != ultima_palabra:
                            print(f"Palabra detectada: {palabra}")
                            pronunciar_palabra(palabra)
                            ultima_palabra = palabra  # Actualizar la última palabra detectada

                    kp_sequence = []  # Reiniciar la secuencia después de la predicción
            
            # Mostrar la palabra traducida en pantalla
            if ultima_palabra:
                cv2.putText(image, f"Traducción: {ultima_palabra}", FONT_POS, FONT, FONT_SIZE, (255, 255, 255), 2)

            cv2.imshow('Traductor de Lengua de Señas', image)

            # Salir al presionar 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    topico = input("Ingresa el tópico para traducir (ejemplo: meses, animales, colores): ")
    traducir_en_tiempo_real(topico)
