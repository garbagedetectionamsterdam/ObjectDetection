from flask import Flask,flash, request
from object_detection.detect_objects import PredictionServer
import os
from threading import Lock
import time
import io


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
		startTime = time.time()*1000.0

		print("received prediction request")
		file_contents = request.stream.read()
		print("read requested image")

		timeAfterRead = time.time()*1000.0

		xml_string = server.detect_objects(io.BytesIO(file_contents))
		print("ran detection")
		timeAfterPrediction = time.time()*1000.0
		print("prediction time " + str(timeAfterPrediction - timeAfterRead))

	return xml_string

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
