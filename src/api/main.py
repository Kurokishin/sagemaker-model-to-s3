import numpy as np
from flask import Flask, request, jsonify
import json
import boto3
import pickle
import os

app = Flask(__name__)

# Inicializa o cliente
s3 = boto3.client('s3')

# Atributo 'Body' do objeto retornado acessa os dados do arquivo
model_file = s3.get_object(Bucket='bucket_name', Key = 'path/to/model.pkl')['Body']

# Carrega o modelo para a aplicação
loaded_model = pickle.load(open(model_file, 'rb'))

@app.route('/')
def index():
    return 'App do Grupo 4!', 200

@app.route('/api/v1/predict', methods=["GET", "POST"])

# Definindo parâmetros de retorno e tratamento de erro
def predict():
    try:
        if request.method == "GET":
            return 'Em desenvolvimento'
        elif request.method == "POST":
            json_data = request.get_json()
            predicao = model_file.predict(np.array([list(json_data.values())]))
            resultado = predicao[0]

            resposta = {"result": int(resultado)}
           
            return jsonify(resposta)

    except Exception as e:
        return jsonify(error=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True)
