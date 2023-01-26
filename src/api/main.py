from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    return 'App do Grupo 4!'

@app.route('/api/v1/predict', methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return 'Em desenvolvimento'

if __name__ == "__main__":
    app.run(debug=True)