import os
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# === Imports conditionnels pour PyTorch (désactivé en prod) ===
# import torch
# import torch.nn as nn
# import torchvision.models as tv_models
# import torchvision.transforms as transforms

BASE_DIR = Path(__file__).resolve().parent.parent

HF_MODEL_URL = "https://huggingface.co/julienlucas/fakefinder/resolve/main/models/best_f191.5_acc91.5_efficientnetv2s_fake_detector_fp16.onnx"
_LOCAL_MODEL = BASE_DIR / "models" / "best_f191.5_acc91.5_efficientnetv2s_fake_detector_fp16.onnx"
_TMP_MODEL = Path("/tmp/fakefinder_model.onnx")


def _get_model_path():
    """Retourne le chemin du modèle : local si disponible, sinon télécharge dans /tmp."""
    if _LOCAL_MODEL.exists():
        return str(_LOCAL_MODEL)
    if _TMP_MODEL.exists():
        return str(_TMP_MODEL)
    # Télécharger depuis HuggingFace (Vercel serverless)
    import urllib.request
    print(f"[ONNX] Downloading model from HuggingFace to {_TMP_MODEL}...")
    urllib.request.urlretrieve(HF_MODEL_URL, str(_TMP_MODEL))
    print(f"[ONNX] Model downloaded ({_TMP_MODEL.stat().st_size / 1e6:.1f} MB)")
    return str(_TMP_MODEL)


ONNX_MODEL_PATH = None  # Resolved lazily
# PTH_MODEL_PATH = str(BASE_DIR / "models" / "best_model_nanobanana_pro.pth")  # Désactivé en prod
REAL_THRESHOLD = 0.7
FAKE_THRESHOLD = 0.7
ENABLE_CAM = False

_session = None
_expected_image_size = None
_input_name = None
# _pytorch_model = None  # Désactivé en prod

def get_onnx_session():
    """Retourne la session ONNX. Le modèle peut avoir ou non les features selon l'export."""
    global _session, _expected_image_size, _input_name, ONNX_MODEL_PATH
    if _session is None:
        if ONNX_MODEL_PATH is None:
            ONNX_MODEL_PATH = _get_model_path()
        _session = ort.InferenceSession(ONNX_MODEL_PATH)
        input_info = _session.get_inputs()[0]
        _input_name = input_info.name
        input_shape = input_info.shape
        if len(input_shape) >= 3 and isinstance(input_shape[-1], int):
            _expected_image_size = input_shape[-1]
        else:
            _expected_image_size = 384
    return _session

def get_expected_image_size():
    if _expected_image_size is None:
        get_onnx_session()
    return _expected_image_size

def get_input_name():
    if _input_name is None:
        get_onnx_session()
    return _input_name

def has_cam_support():
    """Vérifie si le modèle actuel supporte CAM (a une sortie 'features')."""
    session = get_onnx_session()
    output_names = [out.name for out in session.get_outputs()]
    return 'features' in output_names

def draw_bboxes_numpy(pil_image, boxes, labels, bbox_colors, bbox_width=6, font_size=54, text_offset=14, stroke_width=0, padding_left=24, padding_bottom=24, text_padding_left=8, text_padding_bottom=8):
    """Dessine des bboxes sur une image PIL (sans torch)."""
    COLOR_MAP = {"red": (255, 0, 0), "green": (0, 200, 0)}
    FONT_CANDIDATES = [
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    def _color_to_rgb(c):
        if isinstance(c, (tuple, list)) and len(c) >= 3:
            return tuple(int(x) for x in c[:3])
        return COLOR_MAP.get(str(c).lower(), (255, 255, 255))

    def _load_font(size):
        pil_fonts_dir = os.path.join(os.path.dirname(ImageFont.__file__), "fonts")
        candidates = list(FONT_CANDIDATES) + [os.path.join(pil_fonts_dir, "DejaVuSans.ttf")]
        for path in candidates:
            try:
                font = ImageFont.truetype(path, size)
                return font, True
            except Exception:
                continue
        return ImageFont.load_default(), False

    # Dessiner les bboxes avec PIL
    if pil_image.mode != 'RGBA':
        pil_image = pil_image.convert('RGBA')

    img_array = np.array(pil_image)
    draw = ImageDraw.Draw(pil_image)
    font, is_truetype = _load_font(font_size)

    if len(bbox_colors) == 1 and len(boxes) > 1:
        bbox_colors = bbox_colors * len(boxes)

    for (xmin, ymin, xmax, ymax), label, color in zip(boxes, labels, bbox_colors):
        rgb_color = _color_to_rgb(color)
        # Dessiner le rectangle
        for i in range(bbox_width):
            draw.rectangle(
                [xmin + i, ymin + i, xmax - i, ymax - i],
                outline=rgb_color,
                width=1
            )

        # Dessiner le texte
        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        scale = 1
        render_w, render_h = text_w, text_h
        if not is_truetype:
            scale = max(1, int(round(font_size / max(1, text_h))))
            render_w = max(1, int(text_w * scale))
            render_h = max(1, int(text_h * scale))
        x = int(max(0, min(pil_image.size[0] - render_w - 1, xmin + padding_left)))
        if (ymin - render_h - text_offset) >= 0:
            y = int(ymin - render_h - text_offset)
        else:
            y = int(min(pil_image.size[1] - render_h - 1 - padding_bottom, ymax + text_offset - padding_bottom))

        if is_truetype:
            draw.text((x, y), label, fill=rgb_color, font=font, stroke_width=stroke_width, stroke_fill=(0, 0, 0))
        else:
            mask = Image.new("L", (text_w, text_h), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.text((0, 0), label, fill=255, font=font)
            if scale != 1:
                mask = mask.resize((render_w, render_h), resample=Image.NEAREST)
            color_img = Image.new("RGBA", (render_w, render_h), rgb_color + (255,))
            pil_image.paste(color_img, (x, y), mask)

    return pil_image


def softmax_np(x):
    """Softmax numpy (remplace torch.softmax pour éviter la dépendance torch en prod)"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def generate_cam_heatmap(features, pred_idx, original_size):
    """
    Génère une heatmap CAM approximative à partir des features.
    Pour un CAM complet, il faudrait les poids de classification, mais on utilise
    une approximation basée sur l'activation moyenne des features.

    Args:
        features: Activations de la dernière couche convolutive [1, C, H, W]
        pred_idx: Index de la classe prédite (0=fake, 1=real)
        original_size: Taille originale de l'image (width, height)

    Returns:
        Heatmap normalisée [H, W] prête à être superposée sur l'image
    """
    if len(features.shape) != 4 or features.shape[0] != 1:
        return None

    _, C, H, W = features.shape

    cam = np.zeros((H, W), dtype=np.float32)

    for i in range(C):
        feature_map = features[0, i, :, :]
        if pred_idx == 0:
            cam += np.maximum(feature_map, 0)
        else:
            cam += np.maximum(feature_map, 0)

    cam = cam / C
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    cam_resized = Image.fromarray((cam * 255).astype(np.uint8))
    cam_resized = cam_resized.resize(original_size, Image.Resampling.BILINEAR)
    cam_resized = np.array(cam_resized) / 255.0

    return cam_resized

def jet_colormap(x):
    """
    Applique la colormap 'jet' (bleu -> cyan -> vert -> jaune -> rouge) sans matplotlib.
    x doit être entre 0 et 1.
    """
    x = np.clip(x, 0, 1)
    x_4 = x * 4.0
    r = np.clip(np.minimum(x_4 - 1.5, -x_4 + 4.5), 0, 1)
    g = np.clip(np.minimum(x_4 - 0.5, -x_4 + 3.5), 0, 1)
    b = np.clip(np.minimum(x_4 + 0.5, -x_4 + 2.5), 0, 1)
    return np.stack([r, g, b], axis=-1)

def overlay_heatmap(pil_image, heatmap, alpha=0.4):
    """
    Superpose une heatmap sur une image PIL (sans matplotlib).

    Args:
        pil_image: Image PIL originale
        heatmap: Heatmap normalisée [H, W] entre 0 et 1
        alpha: Transparence de la heatmap (0-1)

    Returns:
        Image PIL avec heatmap superposée
    """
    img_array = np.array(pil_image, dtype=np.float32)
    h, w = heatmap.shape[:2]

    if img_array.shape[:2] != (h, w):
        heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = heatmap_resized.resize((w, h), Image.Resampling.BILINEAR)
        heatmap = np.array(heatmap_resized) / 255.0

    heatmap_colored = jet_colormap(heatmap)
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    overlay = (alpha * heatmap_colored + (1 - alpha) * img_array).astype(np.uint8)

    return Image.fromarray(overlay)

def predict_with_gradcam(pil_image, use_cam=False):
    """
    Prédit si une image est fake ou real avec option CAM (Class Activation Mapping).

    Args:
        pil_image: Image PIL à analyser
        use_cam: Si True, génère une heatmap CAM (nécessite modèle exporté avec features)
                 Utilise ONNX_MODEL_PATH_CAM si disponible, sinon ONNX_MODEL_PATH

    Returns:
        result_img, pred_label, conf, real_conf, fake_conf
    """
    session = get_onnx_session()
    image_size = get_expected_image_size()

    w, h = pil_image.size
    original_size = (w, h)

    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_resized = pil_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = (img_array - imagenet_mean) / imagenet_std
    img_array = img_array.transpose(2, 0, 1)
    input_tensor = img_array[np.newaxis, ...]

    input_name = get_input_name()
    output_names = [out.name for out in session.get_outputs()]

    outputs = session.run(output_names, {input_name: input_tensor})

    if 'features' in output_names and 'output' in output_names:
        logits = outputs[output_names.index('output')][0]
    else:
        logits = outputs[0][0]

    probs = softmax_np(logits)
    real_conf = float(probs[1])
    fake_conf = float(probs[0])

    if real_conf >= REAL_THRESHOLD:
        pred_idx = 1
    elif fake_conf >= FAKE_THRESHOLD:
        pred_idx = 0
    else:
        pred_idx = int(np.argmax(probs))

    class_names = ['fake', 'real']
    pred_label = class_names[pred_idx]
    conf = float(probs[pred_idx])

    result_img = pil_image.copy()

    if use_cam and ENABLE_CAM and len(outputs) >= 2:
        try:
            if 'features' in output_names:
                features_idx = output_names.index('features')
                features = outputs[features_idx]

                if len(features.shape) == 4:
                    heatmap = generate_cam_heatmap(features, pred_idx, original_size)
                    if heatmap is not None:
                        result_img = overlay_heatmap(result_img, heatmap, alpha=0.4)
        except Exception:
            pass

    boxes = [[0, 0, w - 1, h - 1]]
    color = "red" if pred_idx == 0 else "green"
    base_font_size = max(32, int(min(w, h) * 0.06))

    result_img = draw_bboxes_numpy(
        result_img,
        boxes=boxes,
        labels=[f"{pred_label[:1].upper()}{pred_label[1:]} {conf * 100:.1f}%"],
        bbox_colors=[color],
        bbox_width=max(8, int(min(w, h) * 0.01)),
        font_size=base_font_size,
    )

    return result_img, pred_label, conf, real_conf, fake_conf

# === GradCAM avec PyTorch (désactivé en prod) ===
# def predict_and_draw_gradcam_bbox(model, image_path, device, class_names=None, keep_top=0.15, alpha=0.8, gamma=2.2, real_threshold=None, fake_threshold=None):
#     """
#     Prédit si une image est fake ou real en utilisant Grad-CAM pour visualiser les zones importantes.
#     Nécessite PyTorch (désactivé en prod).
#     """
#     pass

# if __name__ == "__main__":
#     main()  # Désactivé en prod (nécessite PyTorch)