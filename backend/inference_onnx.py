import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
import torchvision.utils as vutils
import argparse
import os
import onnxruntime as ort

ONNX_MODEL_PATH = "./backend/model/best_model_nanobanana_pro.onnx"
PTH_MODEL_PATH = "./backend/model/best_model_nanobanana_pro.pth"
REAL_THRESHOLD = 0.7
FAKE_THRESHOLD = 0.7
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_session = None
_pytorch_model = None

def get_onnx_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(ONNX_MODEL_PATH)
    return _session

def get_pytorch_model():
    global _pytorch_model
    if _pytorch_model is None:
        state_dict = torch.load(PTH_MODEL_PATH, map_location=torch.device('cpu'))
        model = tv_models.mobilenet_v3_large(weights=None)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=num_features, out_features=2)
        model_state = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                         if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered_dict, strict=False)
        model = model.to(DEVICE)
        model.eval()
        _pytorch_model = model
    return _pytorch_model


def load_mobilenetv3_model(weights_path, num_classes=None):
    """
    Charge un modèle MobileNetV3-Large pré-entraîné depuis torchvision.

    Args:
        weights_path (str): Le chemin du fichier vers les poids du modèle .pth sauvegardés.
        num_classes (int, optional): Nombre de classes. Si fourni, adapte le classifier avant de charger.

    Returns:
        torch.nn.Module: Un modèle MobileNetV3-Large pré-entraîné.
    """

    # Charge le modèle MobileNetV3-Large pré-entraîné sans poids pré-entraînés.
    model = tv_models.mobilenet_v3_large(weights=None)

    # Si num_classes fourni, adapte le classifier
    if num_classes is not None:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)

    # Charge le dictionnaire d'état (poids) depuis le fichier local.
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)

    return model


def draw_bboxes(image_tensor_u8, boxes, labels, bbox_colors, bbox_width=6, font_size=54, text_offset=14, stroke_width=0, padding_left=24, padding_bottom=24):
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

    img_with_boxes = vutils.draw_bounding_boxes(
        image=image_tensor_u8,
        boxes=torch.tensor(boxes, dtype=torch.float),
        colors=bbox_colors,
        width=bbox_width,
    )

    pil_img = Image.fromarray(img_with_boxes.permute(1, 2, 0).cpu().numpy())
    draw = ImageDraw.Draw(pil_img)
    font, is_truetype = _load_font(font_size)

    if len(bbox_colors) == 1 and len(boxes) > 1:
        bbox_colors = bbox_colors * len(boxes)

    for (xmin, ymin, xmax, ymax), label, color in zip(boxes, labels, bbox_colors):
        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        scale = 1
        render_w, render_h = text_w, text_h
        if not is_truetype:
            scale = max(1, int(round(font_size / max(1, text_h))))
            render_w = max(1, int(text_w * scale))
            render_h = max(1, int(text_h * scale))
        x = int(max(0, min(pil_img.size[0] - render_w - 1, xmin + padding_left)))
        if (ymin - render_h - text_offset) >= 0:
            y = int(ymin - render_h - text_offset)
        else:
            y = int(min(pil_img.size[1] - render_h - 1 - padding_bottom, ymax + text_offset - padding_bottom))

        if is_truetype:
            draw.text((x, y), label, fill=_color_to_rgb(color), font=font, stroke_width=stroke_width, stroke_fill=(0, 0, 0))
        else:
            mask = Image.new("L", (text_w, text_h), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.text((0, 0), label, fill=255, font=font)
            if scale != 1:
                mask = mask.resize((render_w, render_h), resample=Image.NEAREST)
            color_img = Image.new("RGBA", (render_w, render_h), _color_to_rgb(color) + (255,))
            pil_img.paste(color_img, (x, y), mask)

    return torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).to(dtype=torch.uint8)


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

    probs = np.exp(logits) / np.sum(np.exp(logits))
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
    img_u8 = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).byte()

    result_tensor = draw_bboxes(
        image_tensor_u8=img_u8,
        boxes=boxes,
        labels=[f"{pred_label[:1].upper()}{pred_label[1:]} {conf * 100:.1f}%"],
        bbox_colors=[color],
        bbox_width=max(8, int(min(w, h) * 0.01)),
        font_size=base_font_size,
    )

    result_img = Image.fromarray(result_tensor.permute(1, 2, 0).cpu().numpy())

    return result_img, pred_label, conf, real_conf, fake_conf

def predict_and_draw_gradcam_bbox(model, image_path, device, class_names=None, keep_top=0.15, alpha=0.8, gamma=2.2, real_threshold=None, fake_threshold=None):
    """
    Prédit si une image est fake ou real en utilisant Grad-CAM pour visualiser les zones importantes.

    Args:
        model: Modèle PyTorch entraîné
        image_path: Chemin vers l'image à prédire
        device: Device PyTorch (cpu/mps/cuda)
        class_names: Liste des noms de classes (ex: ['fake', 'real']). Par défaut: ['fake', 'real']
        keep_top: Proportion de la heatmap à garder pour la bbox (défaut: 0.15)
        alpha: Opacité de l'overlay de la heatmap (défaut: 0.8)
        gamma: Facteur de correction gamma pour renforcer les couleurs (défaut: 2.2)
        real_threshold: Seuil de confiance pour déclarer "real" (défaut: utilise REAL_THRESHOLD global)
        fake_threshold: Seuil de confiance pour déclarer "fake" (défaut: utilise FAKE_THRESHOLD global)
    """
    if class_names is None:
        class_names = ['fake', 'real']

    real_threshold = REAL_THRESHOLD
    fake_threshold = FAKE_THRESHOLD

    model.eval()

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    pil_image = Image.open(image_path).convert("RGB")
    w, h = pil_image.size

    # Trouve la dernière Conv2d
    last_conv = next((m for m in reversed(list(model.features.modules())) if isinstance(m, nn.Conv2d)), None)
    if last_conv is None:
        raise RuntimeError("Impossible de trouver une couche Conv2d dans model.features pour Grad-CAM.")

    # Setup hooks pour Grad-CAM
    activations = None
    gradients = None

    def save_grad(grad):
        nonlocal gradients
        gradients = grad

    def fwd_hook(_module, _inp, out):
        nonlocal activations
        activations = out
        out.register_hook(save_grad)

    handle = last_conv.register_forward_hook(fwd_hook)

    # Forward + backward pour Grad-CAM
    input_tensor = val_transform(pil_image).unsqueeze(0).to(device)
    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0]

    # Utilise les seuils au lieu de argmax simple
    # Assume que class_names[1] est "real" et class_names[0] est "fake"
    real_conf = float(probs[1].item())
    fake_conf = float(probs[0].item())

    # Détermine la prédiction basée sur les seuils
    if real_conf >= real_threshold:
        pred_idx = 1  # real
    elif fake_conf >= fake_threshold:
        pred_idx = 0  # fake
    else:
        # Si aucun seuil n'est atteint, utilise la classe avec la proba la plus élevée
        pred_idx = int(probs.argmax().item())

    pred_label = class_names[pred_idx]
    conf = float(probs[pred_idx].item())

    logits[0, pred_idx].backward()
    handle.remove()

    # Calcule la heatmap Grad-CAM
    # La heatmap montre les régions de l'image qui ont le plus influencé la prédiction
    # Plus c'est rouge/intense, plus cette zone a contribué à la décision
    # Pour "real": zones avec textures naturelles, détails fins, cohérence des ombres/reflets
    # Pour "fake": zones suspectes, textures lisses, incohérences, artéfacts de génération
    cam = (gradients.mean(dim=(2, 3), keepdim=True) * activations).sum(dim=1, keepdim=True)
    cam = F.relu(F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False))[0, 0]
    cam = cam / (cam.max() + 1e-8)

    # Applique gamma et trouve la bbox
    cam_vis = cam.detach().float().cpu().pow(gamma) / (cam.detach().float().cpu().pow(gamma).max() + 1e-8)
    thr = torch.quantile(cam_vis.flatten(), 1.0 - keep_top)
    mask = cam_vis >= thr
    ys, xs = mask.nonzero(as_tuple=True)

    if ys.numel() == 0:
        boxes = [[0, 0, w - 1, h - 1]]
    else:
        xmin_224, xmax_224 = int(xs.min().item()), int(xs.max().item())
        ymin_224, ymax_224 = int(ys.min().item()), int(ys.max().item())
        sx, sy = w / 224.0, h / 224.0
        boxes = [[
            max(0, min(w - 1, int(xmin_224 * sx))),
            max(0, min(h - 1, int(ymin_224 * sy))),
            max(0, min(w - 1, int((xmax_224 + 1) * sx))),
            max(0, min(h - 1, int((ymax_224 + 1) * sy)))
        ]]

    # Overlay additif de la heatmap
    img_f = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
    cam_orig = F.interpolate(cam_vis[None, None, ...], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    overlay = img_f.clone()
    overlay[0] = (overlay[0] + cam_orig * alpha).clamp(0, 1)
    overlay_u8 = (overlay * 255.0).byte()

    # Dessine la bbox et le label
    color = "red" if "fake" in pred_label.lower() else "green"

    # Calcule la taille de police adaptée à la taille de l'image
    base_font_size = max(32, int(min(w, h) * 0.05))

    result = draw_bboxes(
        image_tensor_u8=overlay_u8,
        boxes=boxes,
        labels=[f"{pred_label[:1].upper()}{pred_label[1:]} {conf * 100:.1f}%"],
        bbox_colors=[color],
        bbox_width=max(8, int(min(w, h) * 0.01)),
        font_size=base_font_size,
    )
    return pred_label, conf, boxes[0]


def main():
    parser = argparse.ArgumentParser(description="Prédit si une image est fake ou real")
    parser.add_argument("image_name", help="Nom de l'image dans le répertoire images/")
    args = parser.parse_args()

    model_path = "./models/best_model_nanobanana_pro.pth"

    image_paths = [
        f"./images/{args.image_name}",
        f"./images/fake/{args.image_name}",
        f"./images/real/{args.image_name}",
    ]

    image_path = None
    for path in image_paths:
        if os.path.exists(path):
            image_path = path
            break

    if image_path is None:
        raise FileNotFoundError(f"Image '{args.image_name}' introuvable dans images/")

    trained_model = load_mobilenetv3_model(model_path, num_classes=2)
    trained_model = trained_model.to(DEVICE)
    trained_model.eval()

    predict_and_draw_gradcam_bbox(trained_model, image_path, DEVICE)


if __name__ == "__main__":
    main()