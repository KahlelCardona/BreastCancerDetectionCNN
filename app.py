from flask import Flask, render_template, request
from PIL import Image
from gradcam import predict_with_heatmaps 
import torch

from inference import load_selected_models, predict

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_selected_models(device)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", result=None, error=None)

    file = request.files.get("mammogram")
    if file is None or file.filename == "" or not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        return render_template(
            "index.html", result=None,
            error="Please upload a .jpg, .jpeg, or .png image.",
        )

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return render_template(
            "index.html", result=None,
            error="Could not read that file as an image.",
        )

    result = predict_with_heatmaps(image, models)
    return render_template("index.html", result=result, error=None)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
