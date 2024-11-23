import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

def pronunciar_palabra(palabra):
    engine.say(palabra)
    engine.runAndWait()

if __name__ == "__main__":
    pronunciar_palabra("Hola, mi nombre es Copito")
