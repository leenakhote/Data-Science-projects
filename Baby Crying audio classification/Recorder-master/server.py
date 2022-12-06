import os
from flask import Flask, request, render_template, jsonify
import requests
import json
import recorder as rd
import numpy as np
from scipy.io import wavfile
app = Flask(__name__)


@app.route('/handle_form', methods=['POST'])
def handle_form():
    print("Posted file: {}".format(request.files['file']))
    file = request.files['file']
    files = {'audio': file.read()}
    r = requests.post("http://localhost:7000/predict", files=files)

    if r.ok:
        results = json.loads(r.text)
        # return jsonify({"data": r.text,  "result": "File uploaded"})
        return render_template('result.html', results=results)
    else:
        print(r.status_code, r.reason, r.text)
        return "Error uploading file!"

@app.route('/record_audio', methods=['GET'])
def record_audio():
    result = rd.record()
    if(result == "done"):
        file = os.path.abspath("Recordedfile.wav")
        # file = "/Users/leenakhote/Projects/Recorder/Recordedfile.wav"
        r = requests.get("http://localhost:7000/predictRecorded?file="+str(file))
        if r.ok:
            results = json.loads(r.text)
            # return jsonify({"data": r.text, "result": "File uploaded"})
            return render_template('result.html', results=results)
        else:
            print(r.status_code, r.reason, r.text)
            return "Error Recording file!"

@app.route("/")
def index():
    return render_template("index2.html");


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)


