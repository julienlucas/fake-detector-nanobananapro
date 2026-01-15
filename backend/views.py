import os
import io
import logging
from PIL import Image
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings


logger = logging.getLogger(__name__)

# Import lazy pour éviter les erreurs au démarrage
predict_simple = None

# Configuration LangSmith pour le tracking (si besoin) - désactivé pour éviter les erreurs
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "nanobananapro-fakefinder"

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
        from .inference_onnx import predict_with_gradcam

        file = request.FILES['file']
        contents = file.read()
        if not contents:
            logger.error("Contenu du fichier vide")
            return JsonResponse({"error": "Le contenu du fichier est vide"}, status=400)

        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        result_image, pred_label, conf, real_conf, fake_conf = predict_with_gradcam(pil_image)

        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return JsonResponse({
            "label": pred_label,
            "confidence": conf,
            "real_confidence": real_conf,
            "fake_confidence": fake_conf,
            "image": f"data:image/png;base64,{img_base64}"
        })

    except Exception as e:
        logger.error(f"Erreur lors de l'inférence: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)
