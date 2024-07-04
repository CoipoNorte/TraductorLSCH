# 4. Evaluar Modelo.py

import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from tensorflow.keras.models import load_model
from text_to_speech import text_to_speech

# Definir el código de color ANSI para azul claro
LIGHT_BLUE = '\033[94m'
RESET = '\033[0m'
# Parámetros y configuraciones
MAX_LENGTH_FRAMES = 15
MIN_LENGTH_FRAMES = 5
ROOT_PATH = os.getcwd()
DATA_PATH = os.path.join(ROOT_PATH, "data")
MODELS_PATH = os.path.join(ROOT_PATH, "models")
MODEL_NAME = f"actions_{MAX_LENGTH_FRAMES}.keras"
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5

def mediapipe_detection(image, model):
    """Detecta landmarks en una imagen usando el modelo Holistic de Mediapipe."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image) 
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def there_hand(results):
    """Verifica si se detecta alguna mano en los resultados."""
    return results.left_hand_landmarks or results.right_hand_landmarks

def draw_keypoints(image, results):
    """Dibuja keypoints (puntos característicos) en la imagen con colores diferenciados."""
    draw_landmarks(
        image,
        results.face_landmarks,
        FACEMESH_CONTOURS,
        DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
        DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
    )
    draw_landmarks(
        image,
        results.pose_landmarks,
        POSE_CONNECTIONS,
        DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
        DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
    )
    draw_landmarks(
        image,
        results.left_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
        DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
    )
    draw_landmarks(
        image,
        results.right_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
        DrawingSpec(color=(255, 165, 0), thickness=2, circle_radius=2),
    )

def extract_keypoints(results):
    """Extrae keypoints relevantes de los resultados de la detección."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def get_actions(path):
    """Obtiene las acciones disponibles (clases) desde el directorio especificado."""
    out = []
    for action in os.listdir(path):
        name, ext = os.path.splitext(action)
        if ext == ".h5":
            out.append(name)
    return out

def evaluate_model(model, threshold=0.7):
    """
    Evalúa un modelo de clasificación de acciones en tiempo real desde la cámara.

    Captura vídeo desde la cámara, detecta landmarks usando el modelo Holistic de Mediapipe,
    extrae keypoints relevantes, y clasifica las acciones con el modelo cargado.
    """
    count_frame = 0  # Contador de frames
    kp_sequence, sentence = [], []  # Secuencias de keypoints y sentencias
    actions = get_actions(DATA_PATH)  # Acciones disponibles
    
    # Iniciar modelo Holistic para detectar landmarks faciales, de pose y de manos
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)  # Capturar vídeo desde la cámara
        
        while video.isOpened():  # Bucle principal mientras la cámara esté abierta
            _, frame = video.read()  # Leer frame de video
            
            # Aplicar detección de landmarks usando modelo Holistic
            image, results = mediapipe_detection(frame, holistic_model)
            kp_sequence.append(extract_keypoints(results))  # Extraer keypoints
            
            # Contar frames si hay manos presentes
            if len(kp_sequence) > MAX_LENGTH_FRAMES and there_hand(results):
                count_frame += 1
            else:
                # Si alcanza longitud mínima de frames y hay manos presentes
                if count_frame >= MIN_LENGTH_FRAMES:
                    # Predecir acción usando modelo y secuencia de keypoints
                    res = model.predict(np.expand_dims(kp_sequence[-MAX_LENGTH_FRAMES:], axis=0))[0]
                    
                    # Si confianza de acción predicha es mayor al umbral
                    if res[np.argmax(res)] > threshold:
                        sent = actions[np.argmax(res)]  # Obtener acción
                        sentence.insert(0, sent)  # Insertar al principio de sentencias
                        text_to_speech(sent)  # Convertir acción a voz
                        print(f"{LIGHT_BLUE}{sent}{RESET}")
                        
                    count_frame = 0  # Reiniciar contador de frames
                    kp_sequence = []  # Reiniciar secuencia de keypoints
                    
            
            # Dibujar rectángulo blanco y mostrar sentencia en la imagen
            cv2.rectangle(image, (0, image.shape[0] - 35), (image.shape[1], image.shape[0]), (255, 255, 255), -1)
            if sentence:
                cv2.putText(image, sentence[0], (10, image.shape[0] - 10), FONT, FONT_SIZE, (0, 0, 0))
            
            # Dibujar keypoints en la imagen
            # draw_keypoints(image, results)
            
            cv2.imshow('Traductor', image)  # Mostrar imagen en ventana llamada 'Traductor'
            
            # Esperar 'c' para salir del bucle
            if cv2.waitKey(10) & 0xFF == ord('c'):
                break

# Función principal
def main():
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)
    lstm_model = load_model(model_path)
    evaluate_model(lstm_model)

# Inicio de ejecución
if __name__ == "__main__":
    main()
    print(f"{LIGHT_BLUE}{os.path.basename(__file__)} finalizado...{RESET}")
