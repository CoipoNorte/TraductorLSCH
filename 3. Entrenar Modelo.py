# 3. Entrenar Modelo.py

import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Definir el código de color ANSI para azul claro
LIGHT_BLUE = '\033[94m'
RESET = '\033[0m'
MAX_LENGTH_FRAMES = 15
MODEL_NAME = f"actions_{MAX_LENGTH_FRAMES}.keras"
LENGTH_KEYPOINTS = 1662
NUM_EPOCH = 110

def get_model(output_lenght: int):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(MAX_LENGTH_FRAMES, LENGTH_KEYPOINTS)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_lenght, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_actions(path):
    out = []
    for action in os.listdir(path):
        name, ext = os.path.splitext(action)
        if ext == ".h5":
            out.append(name)
    return out

def get_sequences_and_labels(actions, data_path):
    sequences, labels = [], []
    
    for label, action in enumerate(actions):
        hdf_path = os.path.join(data_path, f"{action}.h5")
        data = pd.read_hdf(hdf_path, key='data')
        
        for _, data_filtered in data.groupby('sample'):
            sequences.append([fila['keypoints'] for _, fila in data_filtered.iterrows()])
            labels.append(label)
            
    return sequences, labels

# Entrenando modelo
def training_model(data_path, model_path):
    # ['palabra_1', 'palabra_2', 'palabra_3'... 'palabra_N']
    actions = get_actions(data_path)
    
    sequences, labels = get_sequences_and_labels(actions, data_path)
    
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH_FRAMES,padding='post', truncating='post', dtype='float32')

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    model = get_model(len(actions))
    model.fit(X, y, epochs=NUM_EPOCH)
    model.summary()
    model.save(model_path)

# Funcion principal
def main():
    root = os.getcwd()
    data_path = os.path.join(root, "data")
    save_path = os.path.join(root, "models")
    model_path = os.path.join(save_path, MODEL_NAME)
    
    training_model(data_path, model_path)

# Inicio de ejecucion
if __name__ == "__main__":
    main()
    print(f"{LIGHT_BLUE}{os.path.basename(__file__)} finalizado...{RESET}")