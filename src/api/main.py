from flask import Flask, jsonify
import os
from endpoint import predicao

app = Flask(__name__)


# Atributo 'Body' do objeto retornado acessa os dados do arquivo
#model_file = s3.get_object(Bucket='modelo-treinado-grupo4', Key = 'modelos/xgboost/output/xgboost-2023-01-27-14-30-50-270/output/model.tar.gz')['Body']
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
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True)