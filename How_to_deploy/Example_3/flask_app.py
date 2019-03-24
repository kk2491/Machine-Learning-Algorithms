from flask import Flask, request, jsonify
from model import get_predicted_class, initClassifierModels

app = Flask(__name__)

@app.route('/classify', methods = ["POST"])

def classify():

	
	jsonRequest = request.get_json()
	print(jsonRequest)
	inputText = jsonRequest["input"]
	print(inputText)
	prediction = get_predicted_class(inputText)
	outputData = {"Classifier_Output":prediction}
	return jsonify(outputData)
	#except:
	#	return "Error in prediction"
