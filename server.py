
#!/usr/local/bin/python3

from simpletransformers.classification import ClassificationModel
from flask import Flask, request

app = Flask(__name__)
app.config["DEBUG"] = True

model = ClassificationModel(
    "roberta", "outputs/checkpoint-183643-epoch-1",
    use_cuda=False
)

@app.route("/", methods=["GET", "POST"])
def home():
    global model

    if request.method == "POST":
        predictions, raw_outputs = model.predict([request.json["message"]])
        return str(predictions)
    elif request.method == "GET":
        return "<h1>Toxcity Filter API</h1>"


app.run()
