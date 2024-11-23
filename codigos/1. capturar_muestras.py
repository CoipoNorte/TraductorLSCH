import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec

PALABRA = "PALBRADEPRUEBA"  # Nombre de la palabra que se está capturando
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capturar_muestras(path, margin_frame=2, min_cant_frames=5):
    create_folder(path)
    cant_sample_exist = len(os.listdir(path))
    count_sample = 0
    frames = []
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        while video.isOpened():
            _, frame = video.read()
            image, results = mediapipe_detection(frame, holistic_model)
            if there_hand(results):
                frames.append(np.asarray(frame))
            else:
                if len(frames) > min_cant_frames + margin_frame:
                    frames = frames[:-margin_frame]
                    output_folder = os.path.join(path, f"sample_{cant_sample_exist + count_sample + 1}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                    count_sample += 1
                frames = []
            draw_keypoints(image, results)
            cv2.imshow(f'Capturando muestras: "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('c'):
                break
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capturar_muestras(f"capturas/{PALABRA}")
