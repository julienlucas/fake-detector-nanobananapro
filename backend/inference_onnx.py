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

BASE_DIR = Path(__file__).resolve().parent
ONNX_MODEL_PATH = str(BASE_DIR / "model" / "best_model_nanobanana_pro_int8.onnx")
# PTH_MODEL_PATH = str(BASE_DIR / "model" / "best_model_nanobanana_pro.pth")  # Désactivé en prod
REAL_THRESHOLD = 0.7
FAKE_THRESHOLD = 0.7

_session = None
# _pytorch_model = None  # Désactivé en prod

def get_onnx_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(ONNX_MODEL_PATH)
    return _session

def draw_bboxes_numpy(pil_image, boxes, labels, bbox_colors, bbox_width=6, font_size=54, text_offset=14, stroke_width=0, padding_left=24, padding_bottom=24):
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

def predict_with_gradcam(pil_image):
    session = get_onnx_session()

    w, h = pil_image.size

    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_resized = pil_image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = (img_array - imagenet_mean) / imagenet_std
    img_array = img_array.transpose(2, 0, 1)
    input_tensor = img_array[np.newaxis, ...]

    outputs = session.run(None, {'input': input_tensor})
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

    boxes = [[0, 0, w - 1, h - 1]]
    color = "red" if pred_idx == 0 else "green"
    base_font_size = max(32, int(min(w, h) * 0.05))

    result_img = draw_bboxes_numpy(
        pil_image.copy(),
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