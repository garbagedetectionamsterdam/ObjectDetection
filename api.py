from flask import Flask,flash, request
from object_detection.detect_objects import PredictionServer
import os
from threading import Lock


app = Flask(__name__)

server = PredictionServer()

@app.route("/")
def hello():
    print("Hello")
    return "Hello World!"



lock = Lock()
@app.route('/predict', methods=['POST'])
def predict():
	with lock:
		print("received prediction request")
		file_contents = request.stream.read()
		print("read requested image")

		#@TODO make this work without a file
		with open("./temp.jpg", "bw") as f:
			f.write(file_contents)
		print("stored requested image")

		xml_string = server.detect_objects('./temp.jpg')
		print("ran detection")

		os.remove("./temp.jpg")
		print("deleted temporary image")

	return xml_string

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
