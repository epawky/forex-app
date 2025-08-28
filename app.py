import os
from flask import Flask, jsonify
from flask import render_template

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify(status="ok")

# For running directly in VS on Windows (F5)
if __name__ == "__main__":
    from waitress import serve
    port = int(os.getenv("PORT", "8000"))
    serve(app, host="0.0.0.0", port=port)

