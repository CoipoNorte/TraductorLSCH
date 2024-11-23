import tensorflow as tf

def verificar_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU disponible:", gpus)
    else:
        print("No se detectó ninguna GPU.")

if __name__ == "__main__":
    verificar_gpu()
