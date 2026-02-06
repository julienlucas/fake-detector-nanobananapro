import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxconverter_common import float16

from torchvision import models as tv_models

MODEL_PATH = "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector.pth"
ONNX_FP32_PATH = "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector.onnx"
ONNX_FP16_PATH = "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector_fp16.onnx"
IMAGE_SIZE = 384
PRUNING_AMOUNT = 0.2  # 20% de pruning global (global_unstructured)
PRUNING_METHOD = "L1"  # Options: "L1", "L2", "random", "structured_L1", "structured_L2"


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


def load_model(pth_path):
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    dropout = checkpoint.get('config', {}).get('dropout', 0.34) if isinstance(checkpoint, dict) else 0.34
    model = EfficientNetV2SModel(dropout=dropout)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'classifier_state_dict' in checkpoint:
        model.backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.head.load_state_dict(checkpoint['classifier_state_dict'], strict=False)
        if 'pool_avg' in checkpoint:
            model.pool_avg.load_state_dict(checkpoint['pool_avg'])
        if 'pool_max' in checkpoint:
            model.pool_max.load_state_dict(checkpoint['pool_max'])
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model



def prune_backbone(model, amount=0.2, method="L1"):
    import torch.nn.utils.prune as prune

    params = [(m, 'weight') for _, m in model.backbone.named_modules() if isinstance(m, nn.Conv2d)]

    if method == "L1":
        pruning_method = prune.L1Unstructured
    elif method == "L2":
        pruning_method = prune.LnUnstructured
        n = 2
    elif method == "random":
        pruning_method = prune.RandomUnstructured
    else:
        pruning_method = prune.L1Unstructured

    print(f"  Pruning {len(params)} couches Conv2d à {amount*100:.0f}% ({method})")

    if method == "L2":
        prune.global_unstructured(params, pruning_method=pruning_method, amount=amount, n=n)
    else:
        prune.global_unstructured(params, pruning_method=pruning_method, amount=amount)

    for module, _ in params:
        prune.remove(module, 'weight')

    return model



def export_onnx_fp32(model, onnx_path):
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    # Détecter la taille des features pour remplacer le pooling adaptatif par fixe
    with torch.no_grad():
        features = model.backbone.features(dummy)
        _, _, h, w = features.shape
        if h == w:
            # Remplacer temporairement le pooling adaptatif par fixe pour l'export
            original_pool_avg = model.pool_avg
            original_pool_max = model.pool_max
            model.pool_avg = nn.AvgPool2d(kernel_size=h, stride=h)
            model.pool_max = nn.MaxPool2d(kernel_size=h, stride=h)
            print(f"  Pooling adaptatif remplacé temporairement par fixe (kernel={h}x{h}) pour l'export")

    try:
        torch.onnx.export(
            model, dummy, onnx_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            verbose=False
        )
        print(f"  Export ONNX FP32: {onnx_path}")
    finally:
        # Restaurer le pooling adaptatif original
        if h == w:
            model.pool_avg = original_pool_avg
            model.pool_max = original_pool_max



def convert_fp16(fp32_path, fp16_path):
    model_fp32 = onnx.load(fp32_path)
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
    onnx.save(model_fp16, fp16_path)
    print(f"  Export ONNX FP16: {fp16_path}")



def evaluate_onnx(onnx_path, data_dirs):
    from torchvision import datasets, transforms
    from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, Precision, Recall

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    supports_batch = input_shape[0] == 'batch_size' or input_shape[0] is None

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_preds, all_labels = [], []

    for data_dir in data_dirs:
        test_dir = os.path.join(data_dir, "test")
        if not os.path.exists(test_dir):
            print(f"  SKIP: {test_dir} introuvable")
            continue

        dataset = datasets.ImageFolder(test_dir, val_transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        print(f"  {os.path.basename(data_dir)}: {len(dataset)} images")

        for images, labels in loader:
            x = images.numpy().astype(np.float32)
            if supports_batch:
                output = session.run(None, {input_name: x})[0]
            else:
                output = np.concatenate([
                    session.run(None, {input_name: x[i:i+1]})[0] for i in range(x.shape[0])
                ], axis=0)
            all_preds.append(torch.from_numpy(np.argmax(output, axis=1)))
            all_labels.append(labels)

    if not all_preds:
        print("  Aucun dataset trouvé.")
        return None

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    class_names = ['fake', 'real']

    cm = MulticlassConfusionMatrix(num_classes=2)(preds, labels)
    f1_macro = F1Score(task="multiclass", num_classes=2, average='macro')(preds, labels)
    f1_cls = F1Score(task="multiclass", num_classes=2, average='none')(preds, labels)
    prec = Precision(task="multiclass", num_classes=2, average='none')(preds, labels)
    rec = Recall(task="multiclass", num_classes=2, average='none')(preds, labels)
    acc = cm.diag().sum().item() / cm.sum().item()
    acc_cls = cm.diag() / cm.sum(axis=1)

    print(f"  Accuracy: {acc:.4f}  |  F1 macro: {f1_macro.item():.4f}")
    for i, name in enumerate(class_names):
        print(f"  {name}: Acc={acc_cls[i].item():.4f}  P={prec[i].item():.4f}  R={rec[i].item():.4f}  F1={f1_cls[i].item():.4f}")

    return acc


def get_size_mb(path):
    total = Path(path).stat().st_size
    data_file = Path(str(path) + ".data")
    if data_file.exists():
        total += data_file.stat().st_size
    return total / (1024 * 1024)


def benchmark(onnx_path, num_runs=100, warmup=10):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    avg, std = np.mean(times), np.std(times)
    return avg, std, 1000 / avg


def cleanup_temp_files():
    """Supprime les fichiers temporaires créés."""
    files_to_remove = [
        "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector.onnx.data",
        "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector.onnx.data",
        "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector.onnx",
    ]

    for file_path in files_to_remove:
        f = Path(file_path)
        if f.exists():
            f.unlink()
            print(f"  Supprimé: {f.name}")
        else:
            print(f"  Non trouvé: {f.name}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")

    os.makedirs("../models", exist_ok=True)

    print("=" * 60)
    print("PRUNING + EXPORT ONNX FP32 + FP16")
    print("=" * 60)

    # 1. Charger + pruner
    print("\n[1/4] Chargement et pruning...")
    model = load_model(MODEL_PATH)
    model = prune_backbone(model, amount=PRUNING_AMOUNT, method=PRUNING_METHOD)

    # 2. Export ONNX FP32
    print("\n[2/4] Export ONNX FP32...")
    export_onnx_fp32(model, ONNX_FP32_PATH)
    del model

    # 3. Conversion FP16
    print("\n[3/4] Conversion FP16...")
    convert_fp16(ONNX_FP32_PATH, ONNX_FP16_PATH)

    # 4. Évaluation + benchmark
    print("\n[4/4] Évaluation et benchmark...")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_base_dir = os.path.join(base_dir, "..", "fake-detector-nanobananapro")
    data_dirs = [
        os.path.join(data_base_dir, "AIvsReal_nanobanana_pro"),
        os.path.join(data_base_dir, "AIvsReal_midjourney_dalle_sd"),
    ]


    print("\n--- FP16 (pruné) ---")
    fp16_acc = evaluate_onnx(ONNX_FP16_PATH, data_dirs)

    # Taille
    fp32_size = get_size_mb(ONNX_FP32_PATH)
    fp16_size = get_size_mb(ONNX_FP16_PATH)
    print(f"\nTaille: FP32 = {fp32_size:.1f}MB | FP16 = {fp16_size:.1f}MB (réduction {((fp32_size - fp16_size) / fp32_size * 100):.0f}%)")

    # Benchmark
    fp32_avg, fp32_std, fp32_fps = benchmark(ONNX_FP32_PATH)
    fp16_avg, fp16_std, fp16_fps = benchmark(ONNX_FP16_PATH)
    print(f"Vitesse FP32: {fp32_avg:.1f}±{fp32_std:.1f}ms ({fp32_fps:.0f} FPS)")
    print(f"Vitesse FP16: {fp16_avg:.1f}±{fp16_std:.1f}ms ({fp16_fps:.0f} FPS)")

    print(f"\nFichier final: {ONNX_FP16_PATH}")

    # Nettoyage des fichiers temporaires
    cleanup_temp_files()
