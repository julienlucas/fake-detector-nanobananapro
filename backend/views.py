import os
import io
import logging
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings


logger = logging.getLogger(__name__)

# Charge .env si présent
load_dotenv()

# Import lazy pour éviter les erreurs au démarrage
predict_simple = None

# Configuration LangSmith pour le tracking (si besoin) - désactivé pour éviter les erreurs
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "fakefinder-nanobananapro"

def _get_langsmith_client():
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return None
    try:
        from langsmith import Client
    except Exception:
        return None
    return Client(
        api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        api_key=api_key,
    )

@csrf_exempt
def index(request):
    """Serve le frontend React"""
    try:
        # Chercher le fichier HTML dans plusieurs chemins possibles
        possible_paths = [
            os.path.join(str(settings.BASE_DIR), 'frontend', 'dist', 'index.html'),
            os.path.join(os.path.dirname(str(settings.BASE_DIR)), 'frontend', 'dist', 'index.html'),
            '/var/task/frontend/dist/index.html',  # Chemin Vercel
            os.path.join(os.getcwd(), 'frontend', 'dist', 'index.html'),
        ]

        for html_path in possible_paths:
            if os.path.exists(html_path) and os.path.isfile(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return HttpResponse(content, content_type='text/html')

        # Si aucun fichier trouvé, retourner une erreur avec les chemins essayés
        error_msg = f"Frontend not found. Tried paths: {possible_paths}\nBASE_DIR: {settings.BASE_DIR}\nCWD: {os.getcwd()}"
        return HttpResponse(error_msg, status=404, content_type='text/plain')
    except Exception as e:
        logger.error(f"Erreur dans index: {str(e)}")
        import traceback
        return HttpResponse(f"Error: {str(e)}\n\n{traceback.format_exc()}", status=500, content_type='text/plain')

@csrf_exempt
@require_http_methods(["POST"])
def inference_api(request):
    """Route d'inférence pour détecter les deepfakes avec Grad-CAM"""
    try:
        import base64
        import time
        from .inference_onnx import predict_with_gradcam

        file = request.FILES['file']
        contents = file.read()
        if not contents:
            logger.error("Contenu du fichier vide")
            return JsonResponse({"error": "Le contenu du fichier est vide"}, status=400)

        client = _get_langsmith_client()
        start_time = datetime.utcnow()
        start_perf = time.perf_counter()

        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        result_image, pred_label, conf, real_conf, fake_conf = predict_with_gradcam(pil_image)

        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        latency_ms = round((time.perf_counter() - start_perf) * 1000, 2)
        label_value = str(pred_label)
        confidence_value = float(conf)
        real_confidence_value = float(real_conf)
        fake_confidence_value = float(fake_conf)

        response = JsonResponse({
            "label": label_value,
            "confidence": confidence_value,
            "real_confidence": real_confidence_value,
            "fake_confidence": fake_confidence_value,
            "image": f"data:image/png;base64,{img_base64}"
        })

        if client:
            try:
                client.create_run(
                    name="fakefinder-inference",
                    run_type="tool",
                    inputs={
                        "filename": getattr(file, "name", ""),
                        "size_bytes": len(contents),
                        "content_type": getattr(file, "content_type", ""),
                    },
                    outputs={
                        "label": label_value,
                        "confidence": confidence_value,
                        "real_confidence": real_confidence_value,
                        "fake_confidence": fake_confidence_value,
                        "latency_ms": latency_ms,
                    },
                    project_name=os.getenv("LANGCHAIN_PROJECT"),
                    start_time=start_time,
                    end_time=datetime.utcnow(),
                )
            except Exception as exc:
                logger.error(f"Erreur LangSmith create_run (success): {str(exc)}")

        return response

    except Exception as e:
        client = _get_langsmith_client()
        if client:
            try:
                client.create_run(
                    name="fakefinder-inference",
                    run_type="tool",
                    inputs={
                        "filename": getattr(file, "name", ""),
                        "size_bytes": len(contents) if "contents" in locals() else 0,
                        "content_type": getattr(file, "content_type", ""),
                    },
                    error=str(e),
                    status="error",
                    project_name=os.getenv("LANGCHAIN_PROJECT"),
                    start_time=start_time if "start_time" in locals() else datetime.utcnow(),
                    end_time=datetime.utcnow(),
                )
            except Exception as exc:
                logger.error(f"Erreur LangSmith create_run (error): {str(exc)}")
        logger.error(f"Erreur lors de l'inférence: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)
