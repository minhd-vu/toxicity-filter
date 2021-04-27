
#!/usr/local/bin/python3

from simpletransformers.classification import ClassificationModel
from flask import Flask, request
from gevent.pywsgi import WSGIServer
import os

app = Flask(__name__)
# app.config["DEBUG"] = True

model = ClassificationModel(
    "roberta", "outputs/model",
    use_cuda=False
)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        predictions, raw_outputs = model.predict([request.json["message"]])
        return str(predictions)
    elif request.method == "GET":
        return "<h1>Toxcity Filter API</h1>"


if __name__ == "__main__":
    http_server = WSGIServer(("", int(os.environ.get("PORT", 5000))), app)
    http_server.serve_forever()
