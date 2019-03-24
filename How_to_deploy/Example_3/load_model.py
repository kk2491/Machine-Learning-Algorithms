# model = 0

def initClassifierModels(app):
	# global model
	model = pickle.load(open("model.pkl", "rb"))
	print("Model Loaded ...!!!")
	return model

def get_predicted_class(data_input):
	inputToClassifier = [data_input]
	output = model.predict(inputToClassifier)
	return output

