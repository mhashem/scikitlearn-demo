from flask import Flask, jsonify, make_response
from flask import request

from fruitclassifier import FruitClassifier

app = Flask(__name__)


@app.route('/version')
def version():
    return jsonify({'version': 0.1}), 200


@app.route('/heartbeat')
def heartbeat():
    return jsonify({'status': 'UP'})


@app.route('/classifier/api/v1/fruit', methods=['POST'])
def classifier():

    if not request.json:
        return make_response(jsonify({'error': 'no content specified'}), 400)

    f1 = request.json['f1']
    f2 = request.json['f2']

    result = fruit_classifier.predict(f1, f2)
    return jsonify({'label': result[0]})


@app.route('/classifier/api/v1/fruit/train', methods=['POST'])
def train():
    fruit_classifier.async_training()
    return jsonify({'state': 'processing'})

if __name__ == '__main__':

    print ' * Loading Classifier services'

    fruit_classifier = FruitClassifier()
    print ' * Loaded FruitClassifier successfully'

    fruit_classifier.load_model()

    app.run(debug=True)

