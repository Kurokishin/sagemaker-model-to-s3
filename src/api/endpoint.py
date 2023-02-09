import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from downloads import downloads

def predicao():

    downloads()
    
    # Recebe o json com os dados a serem enviados para a predição 
    input_data = json.load(open('src/api/input.json'))

    model = tf.keras.models.load_model('model.h5')

    # Previsão
    previsao = model.predict(np.array([list(input_data.values())]))
    
    previsao = list(previsao[0]).index(max(previsao[0]))

    return previsao + 1