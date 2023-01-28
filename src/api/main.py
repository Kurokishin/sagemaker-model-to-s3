from flask import Flask, request, jsonify
import boto3
import pickle

app = Flask(__name__)

# Inicializa o cliente
s3 = boto3.client('s3')

# Atributo 'Body' do objeto retornado acessa os dados do arquivo
model_file = s3.get_object(Bucket='bucket_name', Key = 'path/to/model.pkl')['Body']

# Carrega o modelo para a aplicação
loaded_model = pickle.load(model_file)

@app.route('/')
def index():
    return 'App do Grupo 4!'

@app.route('/api/v1/predict', methods=["GET", "POST"])
def predict():
    try:
        if request.method == "GET":
            return 'Em desenvolvimento'
        elif request.method == "POST":
            json_data = request.get_json()
            # fazer algo com o json_data
            return jsonify(result='Predição feita')
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == "__main__":
    app.run(debug=True)