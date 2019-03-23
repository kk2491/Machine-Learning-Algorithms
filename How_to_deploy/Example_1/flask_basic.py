from flask import Flask 

app = Flask(__name__)
print("__name__ is {}".format(__name__))

@app.route("/")
def index():
	return "<p>Hello World</p>"

if __name__ == "__main__":
	app.run()