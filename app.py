from flask import Flask

app = Flask(__name__)

@app.route("/")
def predict():
    return "Hello World!"