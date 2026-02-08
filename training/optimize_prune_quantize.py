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

USE_CAM = False  # True = export features + output vision CAM, False = inférence simple

MODEL_PATH = "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector.pth"
ONNX_FP16_PATH = "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector_fp16.onnx"
IMAGE_SIZE = 384


ONNX_FP32_PATH = "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector_cam.onnx" if USE_CAM else "../models/best_f191.5_acc91.5_efficientnetv2s_fake_detector.onnx"


class EfficientNetV2SModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
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


class ModelWithFeatures(nn.Module):
    """Wrapper pour exposer features et output pour CAM."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.backbone.features(x)
        avg_p = self.model.pool_avg(features)
        max_p = self.model.pool_max(features)
        x = torch.cat([avg_p, max_p], dim=1)
        output = self.model.head(x)
        return features, output


def load_model(pth_path):
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    dropout = checkpoint.get('config', {}).get('dropout', 0.5) if isinstance(checkpoint, dict) else 0.5
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


def export_onnx_with_cam(model, onnx_path):
    """Exporte le modèle ONNX avec features et output pour CAM."""
    model_cam = ModelWithFeatures(model)
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'features': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    with torch.no_grad():
        features = model.backbone.features(dummy)
        _, _, h, w = features.shape
        if h == w:
            original_pool_avg = model.pool_avg
            original_pool_max = model.pool_max
            model.pool_avg = nn.AvgPool2d(kernel_size=h, stride=h)
            model.pool_max = nn.MaxPool2d(kernel_size=h, stride=h)
            model_cam = ModelWithFeatures(model)
            print(f"  Pooling adaptatif remplacé temporairement par fixe (kernel={h}x{h}) pour l'export")

    try:
        torch.onnx.export(
            model_cam, dummy, onnx_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["features", "output"],
            dynamic_axes=dynamic_axes,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            verbose=False
        )
        if os.path.exists(onnx_path):
            onnx.checker.check_model(onnx_path)
            print(f"  Export ONNX FP32 avec CAM: {onnx_path}")
        else:
            raise RuntimeError(f"Échec de l'export ONNX: fichier non créé")
    except Exception as e:
        print(f"  ERREUR lors de l'export ONNX: {e}")
        raise
    finally:
        if h == w:
            model.pool_avg = original_pool_avg
            model.pool_max = original_pool_max


def export_onnx_fp32(model, onnx_path):
    """Export ONNX avec un seul output (pour extension / inférence simple)."""
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    with torch.no_grad():
        features = model.backbone.features(dummy)
        _, _, h, w = features.shape
        if h == w:
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
        if os.path.exists(onnx_path):
            onnx.checker.check_model(onnx_path)
            print(f"  Export ONNX FP32: {onnx_path}")
    finally:
        if h == w:
            model.pool_avg = original_pool_avg
            model.pool_max = original_pool_max


def convert_fp16(fp32_path, fp16_path):
    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"Fichier FP32 introuvable: {fp32_path}")

    try:
        model_fp32 = onnx.load(fp32_path)

        op_block_list = ['MaxPool', 'AveragePool', 'GlobalAveragePool', 'GlobalMaxPool']

        model_fp16 = float16.convert_float_to_float16(
            model_fp32,
            keep_io_types=True,
            op_block_list=op_block_list
        )
        onnx.save(model_fp16, fp16_path)
        if os.path.exists(fp16_path):
            print(f"  Export ONNX FP16 avec CAM: {fp16_path}")
            print(f"  Opérations exclues de la conversion FP16: {op_block_list}")
        else:
            raise RuntimeError(f"Échec de la conversion FP16: fichier non créé")
    except Exception as e:
        print(f"  ERREUR lors de la conversion FP16: {e}")
        raise


def evaluate_onnx(onnx_path, data_dirs):
    from torchvision import datasets, transforms
    from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, Precision, Recall

    if not os.path.exists(onnx_path):
        print(f"  ERREUR: Fichier introuvable: {onnx_path}")
        return None

    try:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"  ERREUR lors du chargement du modèle ONNX: {e}")
        return None
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    supports_batch = input_shape[0] == 'batch_size' or input_shape[0] is None

    output_names = [out.name for out in session.get_outputs()]
    has_features = 'features' in output_names

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
                outputs = session.run(output_names, {input_name: x})
                if has_features:
                    output = outputs[output_names.index('output')]
                else:
                    output = outputs[0]
            else:
                outputs_list = []
                for i in range(x.shape[0]):
                    outputs = session.run(output_names, {input_name: x[i:i+1]})
                    if has_features:
                        outputs_list.append(outputs[output_names.index('output')])
                    else:
                        outputs_list.append(outputs[0])
                output = np.concatenate(outputs_list, axis=0)
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
    if not os.path.exists(path):
        return 0
    total = Path(path).stat().st_size
    data_file = Path(str(path) + ".data")
    if data_file.exists():
        total += data_file.stat().st_size
    return total / (1024 * 1024)


def benchmark(onnx_path, num_runs=100, warmup=10):
    if not os.path.exists(onnx_path):
        print(f"  ERREUR: Fichier introuvable: {onnx_path}")
        return None, None, None

    try:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_names = [out.name for out in session.get_outputs()]
        dummy = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

        for _ in range(warmup):
            session.run(output_names, {input_name: dummy})

        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            session.run(output_names, {input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)

        avg, std = np.mean(times), np.std(times)
        return avg, std, 1000 / avg
    except Exception as e:
        print(f"  ERREUR lors du benchmark: {e}")
        return None, None, None


def cleanup_temp_files():
    """Supprime les fichiers temporaires créés."""
    files_to_remove = [
        ONNX_FP32_PATH,
        ONNX_FP32_PATH + ".data",
    ]

    for file_path in files_to_remove:
        f = Path(file_path)
        if f.exists():
            f.unlink()
            print(f"  Supprimé: {f.name}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")

    os.makedirs("../models", exist_ok=True)

    print("=" * 60)
    print("EXPORT ONNX (FP32 + FP16)" + (" avec CAM" if USE_CAM else ""))
    print("=" * 60)

    # 1. Charger le modèle
    print("\n[1/6] Chargement du modèle...")
    model = load_model(MODEL_PATH)

    # 2. Pruning
    print("\n[2/6] Pruning du backbone...")
    model = prune_backbone(model, amount=0.2, method="L1")

    # 3. Export ONNX FP32
    print("\n[3/6] Export ONNX FP32" + (" avec CAM..." if USE_CAM else "..."))
    if USE_CAM:
        export_onnx_with_cam(model, ONNX_FP32_PATH)
    else:
        export_onnx_fp32(model, ONNX_FP32_PATH)
    del model

    # 4. Conversion FP16
    print("\n[4/6] Conversion FP16...")
    convert_fp16(ONNX_FP32_PATH, ONNX_FP16_PATH)

    # 5. Évaluation de précision
    print("\n[5/6] Évaluation de précision...")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_base_dir = os.path.join(base_dir, "..", "fake-detector-nanobananapro")
    data_dirs = [
        os.path.join(data_base_dir, "AIvsReal_nanobanana_pro"),
        os.path.join(data_base_dir, "AIvsReal_midjourney_dalle_sd"),
    ]

    print("\n--- FP16 ---")
    fp16_acc = evaluate_onnx(ONNX_FP16_PATH, data_dirs)

    # 6. Benchmark de latence
    print("\n[6/6] Benchmark de latence...")

    if os.path.exists(ONNX_FP32_PATH):
        fp32_size = get_size_mb(ONNX_FP32_PATH)
    else:
        fp32_size = 0

    if os.path.exists(ONNX_FP16_PATH):
        fp16_size = get_size_mb(ONNX_FP16_PATH)
        print(f"\nTaille: FP32 = {fp32_size:.1f}MB | FP16 = {fp16_size:.1f}MB (réduction {((fp32_size - fp16_size) / fp32_size * 100):.0f}%)" if fp32_size > 0 else f"\nTaille: FP16 = {fp16_size:.1f}MB")
    else:
        fp16_size = 0
        print(f"\nTaille: FP32 = {fp32_size:.1f}MB")

    fp32_avg, fp32_std, fp32_fps = benchmark(ONNX_FP32_PATH)
    fp16_avg, fp16_std, fp16_fps = benchmark(ONNX_FP16_PATH)

    if fp32_avg is not None:
        print(f"Latence FP32: {fp32_avg:.1f}±{fp32_std:.1f}ms ({fp32_fps:.0f} FPS)")
    if fp16_avg is not None:
        print(f"Latence FP16: {fp16_avg:.1f}±{fp16_std:.1f}ms ({fp16_fps:.0f} FPS)")

    print(f"\nFichier final: {ONNX_FP16_PATH}")

    # Nettoyage des fichiers temporaires
    cleanup_temp_files()
