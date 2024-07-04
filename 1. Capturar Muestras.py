# 1. Capturar Muestras.py

import os
import cv2
import sys
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from typing import NamedTuple

PALABRA = "caballo"
# Definir el código de color ANSI para azul claro
LIGHT_BLUE = '\033[94m'
RESET = '\033[0m'
# Parametros de la imagen
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

# Crear una carpeta
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image) 
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Existe una mano en pantalla?
def there_hand(results: NamedTuple) -> bool:
    return results.left_hand_landmarks or results.right_hand_landmarks

# Guardar las muestras
def save_frames(frames, output_folder):
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))

# Dibuja los keypoints en la imagen (puntos característicos)
def draw_keypoints(image, results):
    # Dibuja landmarks faciales con colores sugeridos
    draw_landmarks(
        image,
        results.face_landmarks,
        FACEMESH_CONTOURS,
        DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),  # Rojo para faciales
        DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),  # Amarillo para contornos
    )
    
    # Dibuja conexiones de pose con colores sugeridos
    draw_landmarks(
        image,
        results.pose_landmarks,
        POSE_CONNECTIONS,
        DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Verde para pose
        DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),  # Azul para pose (contraste)
    )
    
    # Dibuja conexiones de la mano izquierda con colores sugeridos
    draw_landmarks(
        image,
        results.left_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Verde para mano izquierda
        DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),  # Azul para conexiones de mano izquierda
    )
    
    # Dibuja conexiones de la mano derecha con colores sugeridos
    draw_landmarks(
        image,
        results.right_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),  # Rojo para mano derecha
        DrawingSpec(color=(255, 165, 0), thickness=2, circle_radius=2),  # Naranja para conexiones de mano derecha
    )

# La captura de muestras respecto a la nueva palabra recibida
# - path: ruta de la carpeta de la palabra
# - margin_frame: cantidad de frames que se ignoran al comienzo y al final
# - min_cant_frames: cantidad mínima de frames para cada muestra
def capturar_muestras(path, margin_frame=2, min_cant_frames=5):
    # Crear la carpeta si no existe
    create_folder(path)
    
    # Contar cuántas muestras ya existen en la carpeta
    cant_sample_exist = len(os.listdir(path))
    
    # Contadores y lista para almacenar los frames capturados
    count_sample = 0
    count_frame = 0
    frames = []
    
    # Iniciar el modelo Holistic para detección
    with Holistic() as holistic_model:
        # Abrir la cámara
        video = cv2.VideoCapture(0)
        
        # Loop para capturar video hasta que se presione 'q' para salir
        while video.isOpened():
            # Leer el siguiente frame de la cámara
            _, frame = video.read()
            
            # Realizar detección utilizando el modelo Holistic
            image, results = mediapipe_detection(frame, holistic_model)
            
            # Verificar si hay al menos una mano detectada
            if there_hand(results):
                count_frame += 1
                # Si se supera el margen de frames ignorados al inicio
                if count_frame > margin_frame: 
                    # Mostrar texto de captura en la imagen
                    cv2.putText(image, 'GRABANDO', FONT_POS, FONT, FONT_SIZE, (0, 0, 255))
                    # Almacenar el frame actual en la lista de frames
                    frames.append(np.asarray(frame))
                
            else:
                # Si se detectó menos frames de los necesarios para una muestra válida
                if len(frames) > min_cant_frames + margin_frame:
                    # Eliminar los frames ignorados al final
                    frames = frames[:-margin_frame]
                    # Crear una carpeta para la muestra actual
                    output_folder = os.path.join(path, f"sample_{cant_sample_exist + count_sample + 1}")
                    create_folder(output_folder)
                    # Guardar los frames capturados en la carpeta
                    save_frames(frames, output_folder)
                    # Incrementar el contador de muestras capturadas
                    count_sample += 1
                
                # Reiniciar la lista de frames y el contador de frames
                frames = []
                count_frame = 0
                # Mostrar texto capturar muestras
                cv2.putText(image, 'Buscando Manos...', FONT_POS, FONT, FONT_SIZE, (0, 255, 0))
                
            # Dibujar los keypoints en la imagen
            draw_keypoints(image, results)
            
            # Mostrar la imagen con los keypoints y texto de estado
            cv2.imshow(f'Capturando muestras: "{os.path.basename(path)}"', image)
            
            # Esperar y detectar si se presiona la tecla 'c' chikita para salir del bucle
            if cv2.waitKey(10) & 0xFF == ord('c'):
                break

        # Liberar la cámara y cerrar todas las ventanas de OpenCV
        video.release()
        cv2.destroyAllWindows()

# Funcion principal
def main(word_name):
    # Construir la ruta al subdirectorio frame_actions y luego la ruta completa para word_name (palabra)
    ROOT_PATH = os.getcwd()
    FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
    word_path = os.path.join(FRAME_ACTIONS_PATH, word_name)
    
    # Llamar a la función capturar_muestras con la ruta construida
    capturar_muestras(word_path)

# Inicio de ejecucion
if __name__ == "__main__":
    # Verificar si se proporcionó un argumento
    if len(sys.argv) > 1:
        word_name = sys.argv[1]
    else:
        # Usar palabra por defecto si no se proporciona ninguna
        word_name = PALABRA
    
    # Si la palabra es vacía, usar la palabra por defecto
    if not word_name:
        word_name = PALABRA
    
    main(word_name)
    print(f"{LIGHT_BLUE}{os.path.basename(__file__)} finalizado...{RESET}")
