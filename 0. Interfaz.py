import tkinter as tk
from tkinter import ttk
import subprocess  # Módulo para ejecutar procesos externos

class ProyectoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traductor LSCH - Prototipo VC")
        self.root.geometry("600x400")  # Tamaño inicial de la ventana
        self.root.configure(bg='#0071D5')  # Color de fondo azul para la ventana

        # Marco para la entrada de texto
        self.text_entry_frame = tk.Frame(self.root, bg='#0071D5')
        self.text_entry_frame.pack(pady=30)

        # Etiqueta y entrada de texto
        self.descripcion = tk.Label(self.text_entry_frame, text="Ingrese una nueva palabra:", font=('Arial', 12), bg='#0071D5', fg='white')
        self.descripcion.grid(row=0, column=0, padx=10, pady=10)

        self.palabra_ingresada = tk.Entry(self.text_entry_frame, font=('Arial', 12), bg='white', fg='black', bd=2, relief=tk.FLAT)
        self.palabra_ingresada.grid(row=1, column=0, padx=10, pady=10, ipadx=10, ipady=5, sticky='ew')  # Ajustar el espacio de texto

        # Marco para los botones
        self.botones_frame = tk.Frame(self.root, bg='#0071D5')
        self.botones_frame.pack(pady=20)

        # Botones
        self.btn_capturar = tk.Button(self.botones_frame, text='Capturar Muestras', command=self.capturar, font=('Arial', 12),
                                      bg='#0085FC', fg='white', bd=2, relief=tk.FLAT, width=20, padx=10, pady=5,
                                      cursor='hand2')
        self.btn_capturar.grid(row=0, column=0, padx=10, pady=10)

        self.btn_puntos = tk.Button(self.botones_frame, text='Crear Puntos\nCaracterísticos', command=self.crear_puntos, font=('Arial', 12),
                                    bg='#33A64C', fg='white', bd=2, relief=tk.FLAT, width=20, padx=10, pady=5,
                                    cursor='hand2')
        self.btn_puntos.grid(row=0, column=1, padx=10, pady=10)

        self.btn_entrenar = tk.Button(self.botones_frame, text='Entrenar Modelo', command=self.entrenar, font=('Arial', 12),
                                      bg='#FFBC26', fg='white', bd=2, relief=tk.FLAT, width=20, padx=10, pady=5,
                                      cursor='hand2')
        self.btn_entrenar.grid(row=1, column=0, padx=10, pady=10)

        self.btn_evaluar = tk.Button(self.botones_frame, text='Evaluar Modelo', command=self.evaluar, font=('Arial', 12),
                                     bg='#1CA5B8', fg='white', bd=2, relief=tk.FLAT, width=20, padx=10, pady=5,
                                     cursor='hand2')
        self.btn_evaluar.grid(row=1, column=1, padx=10, pady=10)

    def capturar(self):
        palabra = self.palabra_ingresada.get()
        subprocess.run(["python", "1. Capturar Muestras.py", palabra])

    def crear_puntos(self):
        # Ejecutar el script 2. Crear Puntos Caracteristicos.py
        subprocess.run(["python", "2. Crear Puntos Caracteristicos.py"])

    def entrenar(self):
        # Ejecutar el script 3. Entrenar Modelo.py
        subprocess.run(["python", "3. Entrenar Modelo.py"])

    def evaluar(self):
        # Ejecutar el script 4. Evaluar Modelo.py
        subprocess.run(["python", "4. Evaluar Modelo.py"])

    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ProyectoGUI(root)
    app.start()
