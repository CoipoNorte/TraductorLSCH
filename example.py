import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Interfaz para Proyecto')
        self.setGeometry(100, 100, 800, 600)

        # Botones para cada script
        btn_capturar_muestras = QPushButton('Capturar Muestras', self)
        btn_capturar_muestras.setStyleSheet(
            """
            QPushButton {
                background-color: lightblue;
                border-radius: 10px;
                border: 2px solid gray;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: lightcyan;
            }
            """
        )

        btn_crear_puntos = QPushButton('Crear Puntos Característicos', self)
        btn_crear_puntos.setStyleSheet(
            """
            QPushButton {
                background-color: lightgreen;
                border-radius: 10px;
                border: 2px solid gray;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: lightgreen;
            }
            """
        )

        btn_entrenar_modelo = QPushButton('Entrenar Modelo', self)
        btn_entrenar_modelo.setStyleSheet(
            """
            QPushButton {
                background-color: lightcoral;
                border-radius: 10px;
                border: 2px solid gray;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: lightcoral;
            }
            """
        )

        btn_evaluar_modelo = QPushButton('Evaluar Modelo', self)
        btn_evaluar_modelo.setStyleSheet(
            """
            QPushButton {
                background-color: lightsalmon;
                border-radius: 10px;
                border: 2px solid gray;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: lightsalmon;
            }
            """
        )

        # Etiqueta y campo de texto para ingresar la palabra
        lbl_palabra = QLabel('Ingrese la palabra:')
        self.txt_palabra = QLineEdit()

        # Espacio para la cámara (canvas)
        self.canvas = QLabel()
        self.canvas.setStyleSheet("background-color: lightgray; border: 2px solid gray;")
        self.canvas.setFixedSize(640, 480)
        self.canvas.setAlignment(Qt.AlignCenter)
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder_img.fill(120)
        self.update_canvas(placeholder_img)

        # Layout principal
        vbox_main = QVBoxLayout()

        hbox_word_input = QHBoxLayout()
        hbox_word_input.addWidget(lbl_palabra)
        hbox_word_input.addWidget(self.txt_palabra)

        vbox_main.addLayout(hbox_word_input)
        vbox_main.addWidget(btn_capturar_muestras)
        vbox_main.addWidget(self.canvas)
        vbox_main.addWidget(btn_crear_puntos)
        vbox_main.addWidget(btn_entrenar_modelo)
        vbox_main.addWidget(btn_evaluar_modelo)

        self.setLayout(vbox_main)

        # Conectar eventos a los botones
        btn_capturar_muestras.clicked.connect(self.on_click_capturar_muestras)
        # Conectar los otros botones aquí

        # Iniciar hilo de la cámara
        self.thread = CameraThread()
        self.thread.change_pixmap_signal.connect(self.update_canvas)
        self.thread.start()

    def closeEvent(self, event):
        """
        Detener el hilo de la cámara al cerrar la aplicación.
        """
        self.thread.stop()
        event.accept()

    def update_canvas(self, image):
        """
        Actualizar el contenido del canvas con una nueva imagen de la cámara.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_img)
        self.canvas.setPixmap(pixmap)

    def on_click_capturar_muestras(self):
        """
        Acción a realizar al hacer clic en el botón 'Capturar Muestras'.
        Aquí se podría integrar la lógica para ejecutar el script correspondiente.
        """
        palabra = self.txt_palabra.text()
        print(f'Capturar muestras para la palabra: {palabra}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
