from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return 'App do Grupo 4!'

@app.route('/api/v1/predict', methods=["GET", "POST"])
def predict():
    try:
        if request.method == "GET":
            return 'Em desenvolvimento'
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == "__main__":
    app.run(debug=True)