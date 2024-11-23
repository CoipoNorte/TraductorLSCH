import cv2

def verificar_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return
    print("Cámara detectada correctamente. Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar la imagen.")
            break
        cv2.imshow("Verificación de Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verificar_webcam()
