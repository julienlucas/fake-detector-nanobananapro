import os
import sys
import argparse

import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

import torch.nn as nn
from torchvision import models as tv_models

# Choisir le modèle : commenter l'un des deux (celui qu'on n'utilise pas)
ONNX_MODEL_PATH = "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector_fp16.onnx"
MODEL_PATH = None

IMAGE_SIZE = 384
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_BASE_DIR  = os.path.join(BASE_DIR, "..", "fake-detector-nanobananapro")
DEFAULT_IMAGE_PATH = os.path.join(DATA_BASE_DIR, "images", "real", "GGQ_1723662752223_1723662762427.webp")


class ONNXModelWrapper:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        self.supports_batch = input_shape[0] == 'batch_size' or input_shape[0] is None

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy().astype(np.float32)

        if self.supports_batch:
            output = self.session.run(None, {self.input_name: x})
            return torch.from_numpy(output[0])
        else:
            output = self.session.run(None, {self.input_name: x})
            return torch.from_numpy(output[0])


class EfficientNetV2SModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.34):
        super().__init__()
        backbone = tv_models.efficientnet_v2_s(weights=None)
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs * 2, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone.features(x)
        avg_p = self.pool_avg(features)
        max_p = self.pool_max(features)
        x = torch.cat([avg_p, max_p], dim=1)
        return self.head(x)


def load_efficientnetv2s_model(checkpoint_path, num_classes=2):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    dropout = checkpoint.get('config', {}).get('dropout', 0.34) if isinstance(checkpoint, dict) else 0.34

    model = EfficientNetV2SModel(num_classes=num_classes, dropout=dropout)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint and 'classifier_state_dict' in checkpoint:
            model.backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.head.load_state_dict(checkpoint['classifier_state_dict'], strict=False)
            if 'pool_avg' in checkpoint:
                model.pool_avg.load_state_dict(checkpoint['pool_avg'])
            if 'pool_max' in checkpoint:
                model.pool_max.load_state_dict(checkpoint['pool_max'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict(image_path, model, use_onnx=False):
    image_tensor = load_image(image_path)

    if use_onnx:
        outputs = model(image_tensor)
    else:
        with torch.no_grad():
            outputs = model(image_tensor)

    probs = torch.softmax(outputs, dim=1)[0]
    pred_class = torch.argmax(probs, dim=0).item()
    confidence = probs[pred_class].item()

    class_names = ['fake', 'real']
    return class_names[pred_class], confidence, probs


def main():
    parser = argparse.ArgumentParser(description='Inférence sur une seule image')
    parser.add_argument('image_path', type=str, nargs='?', default=DEFAULT_IMAGE_PATH,
                        help='Chemin vers l\'image à analyser')
    parser.add_argument('--onnx', action='store_true', help='Forcer l\'utilisation du modèle ONNX')
    parser.add_argument('--onnx-path', type=str, default=None, help='Chemin vers le modèle ONNX')
    parser.add_argument('--pth-path', type=str, default=None, help='Chemin vers le modèle PyTorch')

    args = parser.parse_args()

    onnx_path = args.onnx_path or ONNX_MODEL_PATH
    pth_path = args.pth_path or MODEL_PATH

    if args.image_path is None:
        raise ValueError("Chemin d'image requis. Utilisez: python inference.py <chemin_image> ou définissez DEFAULT_IMAGE_PATH")

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image introuvable: {args.image_path}")

    # Choix du modèle : --onnx force ONNX ; sinon celui dont le path est défini (non commenté) et existe
    if args.onnx and onnx_path and os.path.exists(onnx_path):
        use_onnx = True
    elif pth_path and os.path.exists(pth_path):
        use_onnx = False
    elif onnx_path and os.path.exists(onnx_path):
        use_onnx = True
    else:
        raise FileNotFoundError(
            "Aucun modèle disponible. Définir MODEL_PATH (PyTorch) ou ONNX_MODEL_PATH (ONNX) en haut du fichier."
        )

    if use_onnx:
        model = ONNXModelWrapper(onnx_path)
        model_type = "ONNX"
    else:
        model = load_efficientnetv2s_model(pth_path)
        model_type = "PyTorch"

    print(f"Modèle: {model_type}")
    print(f"Image: {args.image_path}")
    print("-" * 50)

    pred_class, confidence, probs = predict(args.image_path, model, use_onnx=use_onnx)

    print(f"Prédiction: {pred_class.upper()}")
    print(f"Confiance: {confidence:.2%}")
    print(f"\nProbabilités:")
    print(f"  fake: {probs[0].item():.2%}")
    print(f"  real: {probs[1].item():.2%}")


if __name__ == "__main__":
    main()
