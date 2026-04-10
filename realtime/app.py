import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
from flask import Flask, request, jsonify, render_template
import io, base64, os, warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load MobileNet
interpreter = tflite.Interpreter(model_path=os.path.join(BASE, "mobilenet_v2_1.0_224_quant.tflite"))
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()
H, W = inp[0]['shape'][1], inp[0]['shape'][2]

# Load labels
with open(os.path.join(BASE, "labels.txt")) as f:
    labels = [l.strip() for l in f]


def run_inference(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((W, H))
    arr = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)
    interpreter.set_tensor(inp[0]['index'], arr)
    interpreter.invoke()
    raw = interpreter.get_tensor(out[0]['index'])[0]
    scale, zp = out[0]['quantization']
    scores = scale * (raw.astype(np.float32) - zp)
    scores = np.exp(scores - np.max(scores))
    scores /= scores.sum()
    return scores

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

@app.route("/llm-benchmark")
def llm_benchmark():
    return render_template("llm-benchmark.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("image", "")
    # Decode base64 image from browser
    img_bytes = base64.b64decode(data.split(",")[1] if "," in data else data)
    scores = run_inference(img_bytes)

    top5 = np.argsort(scores)[::-1][:5]
    results = [{"label": labels[i], "score": round(float(scores[i]) * 100, 1)} for i in top5]

    return jsonify({"top5": results})

if __name__ == "__main__":
    print("\n" + "="*45)
    print("  Real-time AI Vision Server")
    print("  Open your browser at:")
    print("  http://localhost:5000")
    print("="*45 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
