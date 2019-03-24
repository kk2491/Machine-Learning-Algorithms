from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/classifyText', methods = ["POST"])

def predict():

	try:
		jsonRequest = request.get_json()
		inputText = jsonRequest["input"]
		prediction = get_predicted_class(inputText)
		outputData = {"Classifier_Output":prediction}
		return jsonify(outputData)
	except;
		return "Error in prediction"