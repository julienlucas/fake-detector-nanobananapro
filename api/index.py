import os
import sys
import io
import json
import base64
import cgi
from pathlib import Path
from http.server import BaseHTTPRequestHandler

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from PIL import Image
from backend.inference_onnx import predict_with_gradcam, get_onnx_session

# Warmup ONNX au cold start
try:
    get_onnx_session()
except Exception:
    pass

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "3600",
}


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_POST(self):
        try:
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Parse multipart form data
            environ = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(content_length),
            }
            fs = cgi.FieldStorage(
                fp=io.BytesIO(body),
                environ=environ,
                keep_blank_values=True,
            )
            file_item = fs["file"]
            file_data = file_item.file.read()

            if not file_data:
                self._json_response({"error": "Le contenu du fichier est vide"}, 400)
                return

            pil_image = Image.open(io.BytesIO(file_data)).convert("RGB")
            result_image, pred_label, conf, real_conf, fake_conf = predict_with_gradcam(pil_image, use_cam=True)

            img_buffer = io.BytesIO()
            result_image.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            self._json_response({
                "label": str(pred_label),
                "confidence": float(conf),
                "real_confidence": float(real_conf),
                "fake_confidence": float(fake_conf),
                "image": f"data:image/png;base64,{img_base64}",
            })

        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
