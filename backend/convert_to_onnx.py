import os
import warnings
import torch
import torchvision.models as tv_models
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader, QuantType

# Supprimer les warnings de pre-processing (le pre-processing est déjà fait dans CalibrationReader)
warnings.filterwarnings("ignore", message=".*pre-processing.*", category=UserWarning)

MODEL_PATH = "./model/best_model_nanobanana_pro.pth"
ONNX_FP32_PATH = "./model/best_model_nanobanana_pro_fp32.onnx"
ONNX_INT8_PATH = "./model/best_model_nanobanana_pro_int8.onnx"
DEVICE = torch.device("cpu")


def load_mobilenetv3_model(weights_path, num_classes=None):
    model = tv_models.mobilenet_v3_large(weights=None)

    if num_classes is not None:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
    model_state = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items()
                     if k in model_state and model_state[k].shape == v.shape}

    model.load_state_dict(filtered_dict, strict=False)
    return model


def apply_global_pruning(model: torch.nn.Module, amount: float = 0.3) -> torch.nn.Module:
    """Applique le pruning L1 global à toutes les couches Conv2d et Linear."""
    print(f"✂️  Application du pruning L1 GLOBAL (amount={amount})...")

    total_params_before = sum(p.numel() for p in model.parameters())
    nonzero_before = sum((p != 0).sum().item() for p in model.parameters())

    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    print(f"   ✓ {len(parameters_to_prune)} couches identifiées pour le pruning")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    nonzero_after = sum((p != 0).sum().item() for p in model.parameters())
    sparsity = 1 - (nonzero_after / nonzero_before)
    print(f"   ✓ Paramètres: {total_params_before:,}")
    print(f"   ✓ Non-nuls avant: {nonzero_before:,}")
    print(f"   ✓ Non-nuls après: {nonzero_after:,}")
    print(f"   ✓ Sparsité réelle: {sparsity:.2%}")

    return model


def apply_structured_pruning(model: torch.nn.Module, amount: float = 0.3) -> torch.nn.Module:
    """Applique le pruning structuré (par canal) pour réduire la perte de précision."""
    print(f"✂️  Application du pruning STRUCTURÉ (amount={amount})...")

    total_params_before = sum(p.numel() for p in model.parameters())
    nonzero_before = sum((p != 0).sum().item() for p in model.parameters())

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Pruning structuré par canal pour Conv2d
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
        elif isinstance(module, torch.nn.Linear):
            # Pruning non-structuré pour Linear (pas de canaux)
            prune.l1_unstructured(module, name="weight", amount=amount)

    # Rendre le pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.remove(module, "weight")

    nonzero_after = sum((p != 0).sum().item() for p in model.parameters())
    sparsity = 1 - (nonzero_after / nonzero_before)
    print(f"   ✓ Paramètres: {total_params_before:,}")
    print(f"   ✓ Non-nuls avant: {nonzero_before:,}")
    print(f"   ✓ Non-nuls après: {nonzero_after:,}")
    print(f"   ✓ Sparsité réelle: {sparsity:.2%}")

    return model


def convert_to_onnx_fp32(model, onnx_path, input_size=(1, 3, 224, 224)):
    """Étape 1: Convertit le modèle PyTorch FP32 en ONNX FP32."""
    model = model.to(DEVICE)
    model.eval()

    dummy_input = torch.randn(input_size).to(DEVICE)

    print(f"Conversion en ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_shapes=[{0: 'batch_size'}],
            verbose=False
        )
    except Exception as e:
        print(f"Erreur avec opset 18: {e}")
        print("Essai avec opset 14 (plus stable)...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_shapes=[{0: 'batch_size'}],
            verbose=False
        )
    print(f"[1/2] Modèle FP32 exporté : {onnx_path}")


class FakeDetectorCalibrationReader(CalibrationDataReader):
    """Fournit des données de calibration pour la quantification statique."""

    def __init__(self, num_samples=200):
        self.num_samples = num_samples
        self.count = 0

    def get_next(self):
        if self.count >= self.num_samples:
            return None

        self.count += 1
        # Générer des images plus réalistes avec des distributions variées
        # Utiliser une distribution normale centrée plutôt que uniforme
        dummy_image = np.random.normal(0.5, 0.2, (1, 3, 224, 224)).astype(np.float32)
        dummy_image = np.clip(dummy_image, 0.0, 1.0)
        # Normalisation ImageNet standard
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        dummy_image = (dummy_image - mean) / std
        return {"input": dummy_image}


def quantize_onnx_int8(fp32_onnx_path, int8_onnx_path, num_calibration_samples=200):
    """Étape 2: Quantifie le modèle ONNX FP32 en INT8 avec calibration."""
    print(f"[2/2] Quantification INT8...")

    try:
        calibration_reader = FakeDetectorCalibrationReader(num_samples=num_calibration_samples)
        quantize_static(
            model_input=fp32_onnx_path,
            model_output=int8_onnx_path,
            calibration_data_reader=calibration_reader,
            quant_format=ort.quantization.QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            per_channel=True,  # Meilleure précision avec per_channel
        )
    except Exception as e:
        print(f"Erreur avec quantification statique: {e}")
        print("Utilisation de la quantification dynamique (sans calibration)...")
        quantize_dynamic(
            model_input=fp32_onnx_path,
            model_output=int8_onnx_path,
            weight_type=QuantType.QUInt8,
        )
    print(f"[2/2] Modèle INT8 quantifié : {int8_onnx_path}")


def get_model_size_mb(path):
    """Retourne la taille du modèle en MB."""
    p = Path(path)
    total = p.stat().st_size
    data_file = Path(str(path) + ".data")
    if data_file.exists():
        total += data_file.stat().st_size
    return total / (1024 * 1024)


def cleanup_fp32_onnx(fp32_path):
    """Supprime les fichiers ONNX FP32 intermédiaires."""
    fp32_file = Path(fp32_path)
    data_file = Path(str(fp32_path) + ".data")

    for f in [fp32_file, data_file]:
        if f.exists():
            f.unlink()
            print(f"Nettoyage : {f.name} supprimé")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruning-amount", type=float, default=0.2, help="Fraction de poids à pruner (0.0-1.0, défaut: 0.2 pour moins de perte)")
    parser.add_argument("--pruning-type", type=str, choices=["global", "structured"], default="global", help="Type de pruning: 'global' (non-structuré) ou 'structured' (structuré)")
    parser.add_argument("--no-pruning", action="store_true", help="Ignorer le pruning")
    parser.add_argument("--no-quantization", action="store_true", help="Ignorer la quantification")
    args = parser.parse_args()

    print(f"Chargement du modèle PyTorch depuis {MODEL_PATH}...")
    model = load_mobilenetv3_model(MODEL_PATH, num_classes=2)
    model = model.to(DEVICE)
    model.eval()

    if not args.no_pruning:
        if args.pruning_type == "structured":
            model = apply_structured_pruning(model, amount=args.pruning_amount)
        else:
            model = apply_global_pruning(model, amount=args.pruning_amount)

    convert_to_onnx_fp32(model, ONNX_FP32_PATH)

    if not args.no_quantization:
        quantize_onnx_int8(ONNX_FP32_PATH, ONNX_INT8_PATH, num_calibration_samples=200)

        fp32_size = get_model_size_mb(ONNX_FP32_PATH)
        int8_size = get_model_size_mb(ONNX_INT8_PATH)

        cleanup_fp32_onnx(ONNX_FP32_PATH)

        print(f"\n=== Résumé ===")
        print(f"Modèle INT8 : {int8_size:.2f} MB (réduction de {((fp32_size - int8_size) / fp32_size * 100):.1f}% vs FP32)")
        print(f"\nConversion terminée ! Utilisez '{ONNX_INT8_PATH}' pour l'inférence.")
    else:
        fp32_size = get_model_size_mb(ONNX_FP32_PATH)
        print(f"\n=== Résumé ===")
        print(f"Modèle FP32 : {fp32_size:.2f} MB")
        print(f"\nConversion terminée ! Utilisez '{ONNX_FP32_PATH}' pour l'inférence.")
