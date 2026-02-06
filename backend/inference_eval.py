import os
import argparse

import torch
import torch.nn as nn
import torchvision.models as tv_models
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchmetrics.classification import MulticlassConfusionMatrix
from tqdm import tqdm


ONNX_MODEL_PATH = "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector_fp16.onnx"
PTH_MODEL_PATH = None
# Chemin relatif depuis fakefinder-website/backend/ vers fake-detector-nanobananapro/
DATASET_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "fake-detector-nanobananapro")


def load_mobilenetv3_model(weights_path, num_classes=2):
    """Charge un modèle MobileNetV3 depuis un checkpoint PyTorch."""
    model = tv_models.mobilenet_v3_large(weights=None)
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
    model_state = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items()
                     if k in model_state and model_state[k].shape == v.shape}

    model.load_state_dict(filtered_dict, strict=False)
    return model


class ONNXModelWrapper:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) >= 3 and isinstance(input_shape[-1], int):
            self.expected_size = input_shape[-1]
        else:
            self.expected_size = 384

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy().astype(np.float32)
        batch_size = x.shape[0]
        outputs_list = []
        for i in range(batch_size):
            single_input = x[i:i+1]
            output = self.session.run(None, {self.input_name: single_input})
            outputs_list.append(output[0])
        outputs = np.concatenate(outputs_list, axis=0)
        return torch.from_numpy(outputs)


def create_test_dataloader(dataset_path, batch_size=32, image_size=384):
    """Crée un DataLoader pour le dataset de test."""
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_path = os.path.join(dataset_path, "test")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Répertoire de test introuvable: {test_path}")

    test_dataset = ImageFolder(test_path, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    return test_loader, test_dataset


def evaluate_model_on_dataset(model, model_type, dataset_path, dataset_name, device, image_size=384):
    """Évalue un modèle sur un dataset spécifique."""
    batch_size = 1 if model_type == "onnx" else 32
    test_loader, test_dataset = create_test_dataloader(dataset_path, batch_size=batch_size, image_size=image_size)

    all_preds = []
    all_labels = []

    val_loader_with_progress = tqdm(
        test_loader,
        desc=f"Évaluation {dataset_name}",
        leave=False
    )

    with torch.no_grad():
        for batch in val_loader_with_progress:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            if model_type == "onnx":
                batch_outputs = []
                for i in range(images.shape[0]):
                    single_image = images[i:i+1]
                    output = model(single_image)
                    batch_outputs.append(output)
                outputs = torch.cat(batch_outputs, dim=0)
            else:
                outputs = model(images)

            preds = torch.argmax(outputs, 1)

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    num_classes = len(test_dataset.classes)
    confmat = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    cm = confmat(all_preds, all_labels)

    per_class_acc = cm.diag() / cm.sum(axis=1)
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    acc_global = (correct / total) if total else 0.0
    class_names = test_dataset.classes

    print(f"\n--- Rapport de précision par classe ({dataset_name}) ---")
    print(f"Précision globale : {acc_global:.4f}")
    for i, acc in enumerate(per_class_acc):
        print(f"  - Précision pour la classe '{class_names[i]}' : {acc.item():.4f}")
    print(f"\nMatrice de confusion ({dataset_name}):")
    print(cm.cpu().numpy())

    return {
        "dataset_name": dataset_name,
        "accuracy": acc_global,
        "per_class_acc": per_class_acc.cpu().numpy(),
        "confusion_matrix": cm.cpu().numpy(),
        "class_names": class_names,
        "total": total,
        "correct": correct
    }


def main():
    parser = argparse.ArgumentParser(description="Évalue le modèle PyTorch ou ONNX")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Chemin vers le modèle (.pth ou .onnx). Si non spécifié, utilise PTH_MODEL_PATH"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["pth", "onnx", "auto"],
        default="auto",
        help="Type de modèle: 'pth', 'onnx', ou 'auto' (détection automatique)"
    )
    args = parser.parse_args()

    # Déterminer le chemin et le type du modèle
    if args.model_path:
        model_path = args.model_path
        if args.model_type == "auto":
            if model_path.endswith(".onnx"):
                model_type = "onnx"
            elif model_path.endswith(".pth"):
                model_type = "pth"
            else:
                raise ValueError(f"Impossible de détecter le type de modèle depuis l'extension: {model_path}")
        else:
            model_type = args.model_type
    else:
        if ONNX_MODEL_PATH and os.path.exists(ONNX_MODEL_PATH):
            model_path = ONNX_MODEL_PATH
            model_type = "onnx"
        elif PTH_MODEL_PATH and os.path.exists(PTH_MODEL_PATH):
            model_path = PTH_MODEL_PATH
            model_type = "pth"
        else:
            raise FileNotFoundError("Aucun modèle disponible. Définir --model-path ou ONNX_MODEL_PATH/PTH_MODEL_PATH")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")

    # Charger le modèle selon son type
    image_size = 384
    if model_type == "onnx":
        print(f"Chargement du modèle ONNX: {model_path}")
        model = ONNXModelWrapper(model_path)
        image_size = model.expected_size
        print(f"Taille d'image détectée: {image_size}x{image_size}")
    else:
        print(f"Chargement du modèle PyTorch: {model_path}")
        model = load_mobilenetv3_model(model_path, num_classes=2)
        model.eval()

    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Définir les deux datasets
    datasets = [
        ("midjourney", os.path.join(DATASET_BASE_DIR, "AIvsReal_midjourney_dalle_sd")),
        ("nanobanana", os.path.join(DATASET_BASE_DIR, "AIvsReal_nanobanana_pro"))
    ]

    results = []
    for dataset_name, dataset_path in datasets:
        if not os.path.exists(dataset_path):
            print(f"⚠️  Dataset introuvable: {dataset_path}, ignoré")
            continue

        result = evaluate_model_on_dataset(model, model_type, dataset_path, dataset_name, device, image_size=image_size)
        results.append(result)

    # Résumé global
    if len(results) > 1:
        print("Résumé global :")
        total_samples = sum(r["total"] for r in results)
        total_correct = sum(r["correct"] for r in results)
        global_accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0

        print(f"Précision globale combinée: {global_accuracy:.4f}")
        print(f"Total échantillons: {total_samples}")
        print(f"Total corrects: {total_correct}")

        for r in results:
            print(f"\n  {r['dataset_name']}: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")


if __name__ == "__main__":
    main()
