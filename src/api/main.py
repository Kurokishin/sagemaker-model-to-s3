from flask import Flask, jsonify
import os
from endpoint import predicao

app = Flask(__name__)

@app.route('/')
def index():
    return 'App do Grupo 4!', 200

@app.route('/api/v1/predict', methods=["POST"])

# Definindo parâmetros de retorno e tratamento de erro
def predict():
    try:
            predict = predicao()
            resposta = {"result": int(predict)}
           
            return jsonify(resposta)

    except Exception as e:
        return jsonify(error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")
