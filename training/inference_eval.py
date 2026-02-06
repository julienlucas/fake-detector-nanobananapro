import os
import torch
import numpy as np
import random
import onnxruntime as ort
import lightning.pytorch as pl
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset

import utils.helper_utils as helper_utils

DATA_DIRS = [
  os.path.join("../AIvsReal_nanobanana_pro", "test"),
  os.path.join("../AIvsReal_midjourney_dalle_sd", "test"),
]
MAX_PER_DATASET = 500

# Choisir le modèle : définir l'un des deux (l'autre à None)
ONNX_MODEL_PATH = "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector_fp16.onnx"
MODEL_PATH = None

IMAGE_SIZE = 384  # Sera détecté depuis le modèle ONNX si disponible
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class FlatImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.files = []
        for f in sorted(os.listdir(root)):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                self.files.append(os.path.join(root, f))
        self.classes = ["fake", "real"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


class ChestXRayDataModule(pl.LightningDataModule):
    def __init__(self, data_dirs, batch_size=32, image_size=384, max_per_dataset=250):
        super().__init__()
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.max_per_dataset = max_per_dataset
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_dataset = None
        self.class_names = None

    def setup(self, stage=None):
        from torch.utils.data import ConcatDataset

        datasets_list = []
        rng = random.Random(42)

        for data_dir in self.data_dirs:
            if not os.path.isdir(data_dir):
                print(f"SKIP: Répertoire introuvable: {data_dir}")
                continue

            dataset = datasets.ImageFolder(data_dir, self.val_transform)

            # Extraire le nom du dataset depuis le chemin
            normalized_path = os.path.abspath(os.path.normpath(data_dir))
            parts = normalized_path.split(os.sep)
            # Chercher le nom du dataset (avant "test")
            dataset_name = "dataset"
            for part in parts:
                if part in ["AIvsReal_nanobanana_pro", "AIvsReal_midjourney_dalle_sd"]:
                    dataset_name = part
                    break
            if dataset_name == "dataset" and len(parts) >= 2:
                # Prendre le nom du répertoire parent de "test"
                dataset_name = parts[-2] if parts[-1] == "test" else parts[-1]

            total_images = len(dataset)

            if total_images > self.max_per_dataset:
                indices = list(range(total_images))
                rng.shuffle(indices)
                sampled_indices = indices[:self.max_per_dataset]
                dataset = Subset(dataset, sampled_indices)
                print(f"Échantillonnage: {len(sampled_indices)}/{total_images} images de {dataset_name}")
            else:
                print(f"Utilisation de toutes les {total_images} images de {dataset_name}")

            datasets_list.append(dataset)

        if not datasets_list:
            raise FileNotFoundError("Aucun dataset valide trouvé")

        self.val_dataset = ConcatDataset(datasets_list)
        self.class_names = ["fake", "real"]

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

        # Détecter l'index de la sortie des logits (celle qui a 2 dimensions et dont la dernière est petite, genre 2)
        self.logits_index = 0
        outputs_info = self.session.get_outputs()
        for i, out in enumerate(outputs_info):
            shape = out.shape
            # Logits attendus: [batch, 2] ou [batch, num_classes]
            if len(shape) == 2:
                self.logits_index = i
                break

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
            outputs = self.session.run(None, {self.input_name: x})
            return torch.from_numpy(outputs[self.logits_index])
        else:
            batch_size = x.shape[0]
            outputs_list = []
            for i in range(batch_size):
                single_input = x[i:i+1]
                outputs = self.session.run(None, {self.input_name: single_input})
                outputs_list.append(outputs[self.logits_index])
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
    valid_dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    if ONNX_MODEL_PATH and os.path.exists(ONNX_MODEL_PATH):
        use_onnx = True
        model_path = ONNX_MODEL_PATH
        # Détecter la taille d'image attendue depuis le modèle ONNX
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        input_shape = session.get_inputs()[0].shape
        detected_image_size = IMAGE_SIZE
        if len(input_shape) >= 3 and isinstance(input_shape[-1], int):
            detected_image_size = input_shape[-1]
    elif MODEL_PATH and os.path.exists(MODEL_PATH):
        use_onnx = False
        model_path = MODEL_PATH
        detected_image_size = IMAGE_SIZE
    else:
        raise FileNotFoundError(
            "Aucun modèle disponible. Définir ONNX_MODEL_PATH ou MODEL_PATH en haut du fichier."
        )

    dm = ChestXRayDataModule(valid_dirs, batch_size=32, image_size=detected_image_size, max_per_dataset=MAX_PER_DATASET)
    dm.setup()

    num_classes = len(dm.class_names)
    device = torch.device("cpu")

    if use_onnx:
        trained_model = ONNXModelWrapper(model_path)
        model_desc = "ONNX"
    else:
        trained_model = load_efficientnetv2s_model(model_path, num_classes=num_classes)
        trained_model = trained_model.to(device).eval()
        model_desc = "PyTorch"

    from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, Precision, Recall
    from tqdm import tqdm

    all_preds = []
    all_labels_list = []

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
        all_labels_list.append(labels.cpu())

    all_preds = torch.cat(all_preds).flatten().long()
    all_labels = torch.cat(all_labels_list).flatten().long()
    class_names = dm.class_names
    all_preds = all_preds.to(device)
    all_labels = all_labels.to(device)

    from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, Precision, Recall
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

    # Calculer les faux positifs depuis la matrice de confusion
    # FP pour classe i = nombre de fois où on prédit i mais c'est une autre classe
    cm_np = cm.cpu().numpy()
    false_positives = []
    for i in range(num_classes):
        fp = cm_np[i].sum() - cm_np[i][i]  # Total prédictions classe i - TP
        false_positives.append(fp)

    print("--- Rapport de précision par classe ---")
    print(f"Précision globale (Accuracy) : {acc_global:.2%} ({correct}/{total} images)")
    print(f"F1 Score (macro) : {f1_macro_score.item():.4f}")
    print()
    for i, acc in enumerate(per_class_acc):
        fp_pct = (false_positives[i] / total) if total > 0 else 0.0
        print(f"Classe '{class_names[i]}':")
        print(f"  - Accuracy : {acc.item():.2%}")
        print(f"  - Precision : {precision_scores[i].item():.2%}")
        print(f"  - Recall : {recall_scores[i].item():.2%}")
        print(f"  - F1 Score : {f1_per_class_scores[i].item():.4f}")
        print(f"  - Faux Positifs (FP) : {fp_pct:.2%}")
    print()
    helper_utils.plot_confusion_matrix(cm.cpu().numpy(), class_names)


if __name__ == "__main__":
    main()
