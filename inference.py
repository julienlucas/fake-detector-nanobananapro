import torch
from PIL import Image, ImageDraw, ImageFont
import numpy
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
import torchvision.utils as vutils

import utils.helper_utils as helper_utils


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


def draw_bboxes(image_tensor_u8, boxes, labels, bbox_colors, bbox_width=6, font_size=54, text_offset=14, stroke_width=0):
    COLOR_MAP = {"red": (255, 0, 0), "green": (0, 200, 0)}
    FONT_CANDIDATES = [
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/System/Library/Fonts/Supplemental/Tahoma.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    def _color_to_rgb(c):
        if isinstance(c, (tuple, list)) and len(c) >= 3:
            return tuple(int(x) for x in c[:3])
        return COLOR_MAP.get(str(c).lower(), (255, 255, 255))

    def _load_font(size):
        for path in FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(path, size)
                return font
            except Exception:
                continue
        raise RuntimeError(f"Impossible de charger une police TrueType. Taille demandée: {size}")

    img_with_boxes = vutils.draw_bounding_boxes(
        image=image_tensor_u8,
        boxes=torch.tensor(boxes, dtype=torch.float),
        colors=bbox_colors,
        width=bbox_width,
    )

    pil_img = Image.fromarray(img_with_boxes.permute(1, 2, 0).cpu().numpy())
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(font_size)

    if len(bbox_colors) == 1 and len(boxes) > 1:
        bbox_colors = bbox_colors * len(boxes)

    for (xmin, ymin, xmax, ymax), label, color in zip(boxes, labels, bbox_colors):
        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        x = int(max(0, min(pil_img.size[0] - 1, xmin)))
        y = int(ymin - text_h - text_offset) if (ymin - text_h - text_offset) >= 0 else int(min(pil_img.size[1] - text_h - 1, ymax + text_offset))

        draw.text((x, y), label, fill=_color_to_rgb(color), font=font, stroke_width=stroke_width, stroke_fill=(0, 0, 0))

    return torch.from_numpy(numpy.array(pil_img)).permute(2, 0, 1).to(dtype=torch.uint8)


# Seuil de confiance pour déclarer une image comme "real" ou "fake" (0.7 = 70%)
REAL_THRESHOLD = 0.7
FAKE_THRESHOLD = 0.7

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
    img_f = torch.from_numpy(numpy.array(pil_image)).permute(2, 0, 1).float() / 255.0
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
    helper_utils.display_images(processed_image=result, figsize=(7, 7))
    return pred_label, conf, boxes[0]


model_path = "./best_model_nanobanana.pth"
image_path = './images/real/IMG_6396.jpg'

trained_model = load_mobilenetv3_model(model_path, num_classes=2)
trained_model = trained_model.to(DEVICE)
trained_model.eval()

predict_and_draw_gradcam_bbox(trained_model, image_path, DEVICE)
