
# Initialze the vectorizer 
vectorizer = TfidVectorizer(encoding = 'latin1', max_df = 0.5, stop_words = 'english', lowercase = True, ngram_range = (1, 3))

# Vectorize the input data with TF-IDF vectorizer
counts = vectorizer.fit_transform(contents)

# Export the vectorizer
joblib.dump(vectorizer, "tfvect.pkl")

# Initialize the classifier
classifier = MultinomialNB()

# Train the classifier with training data (Contents, Labels)
classifier.fit(counts, targets)

# Export the model
with open("model", "wb") as f:
	pickle.dump(classifier, f)

