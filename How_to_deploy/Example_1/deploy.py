import pickle
import flask
from flask import Flask 
from flask import request

app = Flask(__name__)
print(app)

'''
@app.route('/hello', methods=["POST"])

def index():
	name = request.get_json()["name"]
	return "Hello " + name
'''

model = pickle.load(open("model.pkl", "rb"))

#@app.route("/predict", methods = ["POST"])
@app.route('/predict', methods=['POST'])
def predict():
	feature_array = request.get_json()["feature_array"]

	prediction = model.predict([feature_array]).tolist()

	response = {}
	response["predictions"] = prediction

	return flask.jsonify(response)

if __name__ == "__main__":
	app.run()
