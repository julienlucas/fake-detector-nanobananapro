import os
import random
import torch
import numpy as np
import onnxruntime as ort
import lightning.pytorch as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

import utils.helper_utils as helper_utils

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_base_dir = os.path.join(base_dir, "..", "fake-detector-nanobananapro")
DATA_DIRS = [
    os.path.join(data_base_dir, "AIvsReal_nanobanana_pro"),
    os.path.join(data_base_dir, "AIvsReal_midjourney_dalle_sd"),
]
MAX_PER_DATASET = 250

# Choisir le modèle : définir l'un des deux (l'autre à None)
ONNX_MODEL_PATH = "../models/best_mobilenetV3_fake_detector_int8.onnx"
MODEL_PATH = None

IMAGE_SIZE = 384


class ChestXRayDataModule(pl.LightningDataModule):
    def __init__(self, data_dirs, max_per_dataset=250, batch_size=32, image_size=384):
        super().__init__()
        self.data_dirs = data_dirs
        self.max_per_dataset = max_per_dataset
        self.batch_size = batch_size
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_dataset = None
        self.class_names = None

    def setup(self, stage=None):
        rng = random.Random(42)
        parts = []
        for data_dir in self.data_dirs:
            test_dir = os.path.join(data_dir, "test")
            if not os.path.exists(test_dir):
                continue
            ds = datasets.ImageFolder(test_dir, self.val_transform)
            n = min(self.max_per_dataset, len(ds))
            indices = rng.sample(range(len(ds)), n)
            parts.append(Subset(ds, indices))
            if self.class_names is None:
                self.class_names = ds.classes
        if not parts:
            raise FileNotFoundError(f"Aucun répertoire test trouvé dans {self.data_dirs}")
        self.val_dataset = ConcatDataset(parts)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False
        )


class ONNXModelWrapper:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        self.supports_batch = input_shape[0] == 'batch_size' or input_shape[0] is None

        if len(input_shape) >= 3 and isinstance(input_shape[-1], int):
            self.expected_size = input_shape[-1]
        else:
            self.expected_size = None

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
            try:
                output = self.session.run(None, {self.input_name: single_input})
                outputs_list.append(output[0])
            except Exception as e:
                single_input_flat = single_input.reshape(1, -1) if len(single_input.shape) > 2 else single_input
                output = self.session.run(None, {self.input_name: single_input_flat})
                outputs_list.append(output[0])
        outputs = np.concatenate(outputs_list, axis=0)
        return torch.from_numpy(outputs)


import torch.nn as nn
from torchvision import models as tv_models

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




def main():
    for d in DATA_DIRS:
        test_dir = os.path.join(d, "test")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Répertoire test introuvable: {test_dir}")

    if ONNX_MODEL_PATH and os.path.exists(ONNX_MODEL_PATH):
        use_onnx = True
        model_path = ONNX_MODEL_PATH
        temp_wrapper = ONNXModelWrapper(model_path)
        image_size = temp_wrapper.expected_size if temp_wrapper.expected_size else IMAGE_SIZE
        trained_model = temp_wrapper
        model_desc = "ONNX"
        if temp_wrapper.expected_size and temp_wrapper.expected_size != IMAGE_SIZE:
            print(f"Modèle ONNX attend {temp_wrapper.expected_size}x{temp_wrapper.expected_size}, utilisation de cette taille")
    elif MODEL_PATH and os.path.exists(MODEL_PATH):
        use_onnx = False
        model_path = MODEL_PATH
        image_size = IMAGE_SIZE
    else:
        raise FileNotFoundError(
            "Aucun modèle disponible. Définir ONNX_MODEL_PATH ou MODEL_PATH en haut du fichier."
        )

    dm = ChestXRayDataModule(DATA_DIRS, max_per_dataset=MAX_PER_DATASET, batch_size=32, image_size=image_size)
    dm.setup()

    num_classes = len(dm.class_names)
    device = torch.device("cpu")

    if not use_onnx:
        trained_model = load_efficientnetv2s_model(model_path, num_classes=num_classes)
        trained_model = trained_model.to(device).eval()
        model_desc = "PyTorch"

    from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, Precision, Recall
    from tqdm import tqdm

    all_preds = []
    all_labels = []

    val_loader_with_progress = tqdm(
        dm.val_dataloader(),
        desc=f"Évaluation du modèle {model_desc}",
        leave=False
    )

    for batch in val_loader_with_progress:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        if use_onnx:
            outputs = trained_model(images)
        else:
            with torch.no_grad():
                outputs = trained_model(images)
        preds = torch.argmax(outputs, 1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    confmat = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    cm = confmat(all_preds, all_labels)

    f1_macro = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average='none').to(device)
    precision = Precision(task="multiclass", num_classes=num_classes, average='none').to(device)
    recall = Recall(task="multiclass", num_classes=num_classes, average='none').to(device)

    f1_macro_score = f1_macro(all_preds, all_labels)
    f1_per_class_scores = f1_per_class(all_preds, all_labels)
    precision_scores = precision(all_preds, all_labels)
    recall_scores = recall(all_preds, all_labels)

    per_class_acc = cm.diag() / cm.sum(axis=1)
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    acc_global = (correct / total) if total else 0.0
    class_names = dm.class_names

    print("--- Rapport de précision par classe ---")
    print(f"Précision globale (Accuracy) : {acc_global:.4f}")
    print(f"F1 Score (macro) : {f1_macro_score.item():.4f}")
    print()
    for i, acc in enumerate(per_class_acc):
        print(f"Classe '{class_names[i]}':")
        print(f"  - Accuracy : {acc.item():.4f}")
        print(f"  - Precision : {precision_scores[i].item():.4f}")
        print(f"  - Recall : {recall_scores[i].item():.4f}")
        print(f"  - F1 Score : {f1_per_class_scores[i].item():.4f}")
    print()

    helper_utils.plot_confusion_matrix(cm.cpu().numpy(), class_names)


if __name__ == "__main__":
    main()
