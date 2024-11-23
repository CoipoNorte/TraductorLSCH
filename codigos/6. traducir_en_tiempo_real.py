import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from tensorflow.keras.models import load_model
from text_to_speech import pronunciar_palabra

# Configuración
TAMANO_OBJETIVO = (128, 128)
MAX_LENGTH_FRAMES = 30
THRESHOLD = 0.7
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_POS = (10, 30)
FONT_SIZE = 0.8
DATA_PATH = 'datos/'

# Cargar el modelo entrenado
MODELS_PATH = 'modelos/'
MODEL_NAME = 'sign_language_model.h5'
model_path = os.path.join(MODELS_PATH, MODEL_NAME)
modelo = load_model(model_path)

# Obtener las clases
def get_actions(data_path):
    return sorted([action for action in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, action))])

actions = get_actions(DATA_PATH)
ultima_palabra = ""  # Última palabra para evitar repeticiones

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

# Traducir en tiempo real con el modelo de CNN basado en imágenes
def traducir_en_tiempo_real(model):
    global ultima_palabra
    cap = cv2.VideoCapture(0)
    kp_sequence = []

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

                # Limitar la longitud de la secuencia
                if len(kp_sequence) == MAX_LENGTH_FRAMES:
                    # Realizar predicción en cada frame de la secuencia
                    predicciones = [model.predict(np.expand_dims(f, axis=0))[0] for f in kp_sequence]
                    
                    # Votación mayoritaria para obtener la predicción final
                    votos = np.array([np.argmax(pred) for pred in predicciones])
                    resultado = np.bincount(votos).argmax()
                    confianza = np.mean([pred[resultado] for pred in predicciones])
                    
                    # Validar confianza y actualizar la traducción
                    if confianza > THRESHOLD:
                        palabra = actions[resultado]
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
    traducir_en_tiempo_real(modelo)
