def initClassifierModels(app):
	
	global vectorizer
	vectorizer = jobib.load("tfvect.pkl")

	objects = []
	with (open("nbmodel.py", "rb")) as openfile:
		while True:
			try:
				objects.append(pickle.load(openfile))
			except EOFError:
				break

	global classifier 
	classifier = objects[0]


def get_predicted_class(text_input):
	inputToClassifier = [text_input]
	counts = vectorizer.transform(inputToClassifier)
	output = classifier.predict(counts)
	return output[0]

