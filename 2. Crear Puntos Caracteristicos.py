# 2. Crear Puntos Caracteristicos.py

import os
import cv2
import pandas as pd
import numpy as np
from mediapipe.python.solutions.holistic import Holistic

# Definir el código de color ANSI para azul claro
LIGHT_BLUE = '\033[94m'
RESET = '\033[0m'
ROOT_PATH = os.getcwd()
DATA_PATH = os.path.join(ROOT_PATH, "data")
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image) 
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Extraer keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Obtener keypoints de la muestra
def get_keypoints(model, path):
    kp_seq = np.array([])
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        frame = cv2.imread(img_path)
        _, results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq = np.concatenate([kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])
    return kp_seq

# Insertar secuencia de keyponints, Retorna el mismo DataFrame pero con los keypoints de la muestra agregados
def insert_keypoints_sequence(df, n_sample: int, kp_seq):
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'frame': frame + 1,'keypoints': [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints])
    return df

# Crear keypoints para una palabrota, Recorre la carpeta de frames de la palabra y guarda sus keypoints en `save_path`
def create_keypoints(frames_path, save_path):
    data = pd.DataFrame([])
    
    with Holistic() as model_holistic:
        for n_sample, sample_name in enumerate(os.listdir(frames_path), 1):
            sample_path = os.path.join(frames_path, sample_name)
            keypoints_sequence = get_keypoints(model_holistic, sample_path)
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)

    data.to_hdf(save_path, key="data", mode="w")

# Funcion principal
def main():
    words_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH)
    
    # Generar Todos los Keypoints
    for word_name in os.listdir(words_path):
        word_path = os.path.join(words_path, word_name)
        hdf_path = os.path.join(DATA_PATH, f"{word_name}.h5")
        print(f'{LIGHT_BLUE}Creando keypoints: "{word_name}" :({RESET}')
        create_keypoints(word_path, hdf_path)
        print(f"{LIGHT_BLUE}Keypoints Creados :D{RESET}")

    # Ejemplo (para una palabra en especifico)
    # word_name = "PALABRA"
    # word_path = os.path.join(words_path, word_name)
    # hdf_path = os.path.join(data_path, f"{word_name}.h5")
    # create_keypoints(word_path, hdf_path)
    # print(f"Keypoints creados!")

# Inicio de ejecucion
if __name__ == "__main__":
    main()
    print(f"{LIGHT_BLUE}{os.path.basename(__file__)} finalizado...{RESET}")