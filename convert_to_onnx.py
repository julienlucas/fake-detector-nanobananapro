import os
import torch
import torchvision.models as tv_models
import torch.nn as nn

MODEL_PATH = "./backend/model/best_model_nanobanana_pro.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_mobilenetv3_model(weights_path, num_classes=None):
    model = tv_models.mobilenet_v3_large(weights=None)

    if num_classes is not None:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model_state = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items()
                     if k in model_state and model_state[k].shape == v.shape}

    model.load_state_dict(filtered_dict, strict=False)
    return model

def convert_to_onnx():
    """Convertit le modèle PyTorch en format ONNX"""

    # Chemin de sortie pour le modèle ONNX
    onnx_path = MODEL_PATH.replace('.pth', '.onnx')

    print(f"Chargement du modèle PyTorch depuis {MODEL_PATH}...")
    model = load_mobilenetv3_model(MODEL_PATH, num_classes=2)
    model = model.to(DEVICE)
    model.eval()

    # Créer un exemple d'input (taille d'image attendue: 224x224)
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

    print(f"Conversion en ONNX...")
    print(f"Le modèle sera sauvegardé dans: {onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        verbose=False
    )

    print(f"✅ Conversion réussie! Modèle ONNX sauvegardé: {onnx_path}")

    pytorch_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB

    print(f"\nTaille du modèle PyTorch: {pytorch_size:.2f} MB")
    print(f"Taille du modèle ONNX: {onnx_size:.2f} MB")
    print(f"Réduction: {pytorch_size - onnx_size:.2f} MB ({((pytorch_size - onnx_size) / pytorch_size * 100):.1f}%)")

if __name__ == "__main__":
    convert_to_onnx()
