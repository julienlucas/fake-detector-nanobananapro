import os

import torch
import numpy as np
import onnxruntime as ort

import utils.helper_utils as helper_utils
from train import ChestXRayDataModule

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_base_dir = os.path.join(base_dir, "..", "fake-detector-nanobananapro")
DATA_DIR = os.path.join(data_base_dir, "AIvsReal_nanobanana_pro")
ONNX_MODEL_PATH = "../models/best_mobilenetV3_nanobanana_pro_int8.onnx"


# def load_quantized_model(checkpoint_path):
#     ckpt = torch.load(checkpoint_path, map_location="cpu")

#     if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
#         quantized_state_dict = ckpt["model_state_dict"]
#     else:
#         quantized_state_dict = ckpt

#     model_fp32 = helper_utils.resnet18_qat_ready_pretrained(num_classes=3, use_quant_stubs=False)

#     base_weights_path = "../models/best_90_resnet18_chest_xray_classifier_weights_pruned.pth"
#     base_weights = torch.load(base_weights_path, map_location="cpu")
#     model_fp32.load_state_dict(base_weights, strict=False)

#     wrapped_model = QATWrapper(model_fp32)
#     qat_model = prepare_qat(wrapped_model, backend="qnnpack")

#     qat_model.eval()
#     int8_model = torch.quantization.convert(qat_model)

#     int8_model.load_state_dict(quantized_state_dict, strict=False)
#     int8_model.eval()

#     return int8_model

class ONNXModelWrapper:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.eval()

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


def main():
    if not os.path.exists(ONNX_MODEL_PATH):
        raise FileNotFoundError(f"Fichier introuvable: {ONNX_MODEL_PATH}")

    abs_data_dir = os.path.abspath(DATA_DIR)
    if not os.path.exists(abs_data_dir):
        raise FileNotFoundError(
            f"Répertoire dataset introuvable: {DATA_DIR}\n"
            f"Chemin absolu: {abs_data_dir}\n"
            f"Veuillez créer le répertoire ou modifier DATA_DIR dans le script."
        )

    dm = ChestXRayDataModule(DATA_DIR, batch_size=32)
    dm.setup()
    model = ONNXModelWrapper(ONNX_MODEL_PATH)

    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    trained_model = model
    trained_model.eval()
    device = torch.device("cpu")
    trained_model = trained_model.to(device)

    from torchmetrics.classification import MulticlassConfusionMatrix
    from tqdm import tqdm

    all_preds = []
    all_labels = []

    val_loader_with_progress = tqdm(
        dm.val_dataloader(),
        desc="Évaluation du modèle",
        leave=False
    )

    with torch.no_grad():
        for batch in val_loader_with_progress:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = trained_model(images)
            preds = torch.argmax(outputs, 1)

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    num_classes = len(dm.val_dataset.classes)
    confmat = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    cm = confmat(all_preds, all_labels)

    per_class_acc = cm.diag() / cm.sum(axis=1)
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    acc_global = (correct / total) if total else 0.0
    class_names = dm.val_dataset.classes

    print("--- Rapport de précision par classe ---")
    print(f"Précision globale : {acc_global:.4f}")
    for i, acc in enumerate(per_class_acc):
        print(f"  - Précision pour la classe '{class_names[i]}' : {acc.item():.4f}")
    print()

    helper_utils.plot_confusion_matrix(cm.cpu().numpy(), class_names)


if __name__ == "__main__":
    main()
