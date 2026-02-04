import copy
import os
import torch
import torch.nn as nn
import torch.ao.quantization as aoq
from tqdm import tqdm
from train import ChestXRayDataModule, data_dir
import utils.helper_utils as helper_utils


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Périphérique utilisé: {DEVICE}")


def fuse_model_inplace(model: nn.Module) -> nn.Module:
    """
    Applique récursivement une fusion optimale à :
      Conv+BN+ReLU, Conv+BN, Conv+ReLU, Linear+ReLU
    Fusionne uniquement les modules *adjacents* dans les blocs nn.Sequential.
    Modifie `model` en place et retourne la *même instance*.
    """
    # Itérer sur les enfants nommés du modèle
    for _, child in model.named_children():
        # Récursion d'abord
        # Appliquer récursivement la fusion optimale à l'enfant
        fuse_model_inplace(child)

        # Puis scanner cet enfant s'il est un Sequential
        # Vérifier si l'enfant est un Sequential et a au moins 2 couches
        if isinstance(child, nn.Sequential) and len(child) >= 2:
            # Le pliage BN préfère eval ; ne pas muter l'état externe de manière permanente
            # Obtenir l'état d'entraînement de l'enfant
            was_training = child.training
            # Mettre l'enfant en mode eval
            child.eval()
            i = 0
            # Itérer sur les couches de l'enfant - 1
            while i < len(child) - 1:
                # Obtenir les deux couches adjacentes à i et i + 1
                a, b = child[i], child[i+1]
                # Vérifier si la troisième couche (i+2 < len(child)) existe
                if (i+2 < len(child)):
                    # Obtenir la troisième couche
                    c = child[i+2]
                else:
                    # set the third layer to None
                    c = None

                # Conv + BN + ReLU
                # Vérifier si la première couche est un Conv2d, la deuxième est un BatchNorm2d, et la troisième est un ReLU
                if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d) and isinstance(c, nn.ReLU):
                    # Essayer de fusionner les trois couches
                    torch.quantization.fuse_modules(child, [str(i), str(i+1), str(i+2)], inplace=True)
                    i += 3
                    continue
                # Conv + BN
                # Vérifier si la première couche est un Conv2d et la deuxième est un BatchNorm2d
                if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                    # Essayer de fusionner les deux couches
                    torch.quantization.fuse_modules(child, [str(i), str(i+1)], inplace=True)
                    i += 2
                    continue
                # Conv + ReLU
                # Vérifier si la première couche est un Conv2d et la deuxième est un ReLU
                if isinstance(a, nn.Conv2d) and isinstance(b, nn.ReLU):
                    # Essayer de fusionner les deux couches
                    torch.quantization.fuse_modules(child, [str(i), str(i+1)], inplace=True)
                    i += 2
                    continue
                # Linear + ReLU
                # Vérifier si la première couche est un Linear et la deuxième est un ReLU
                if isinstance(a, nn.Linear) and isinstance(b, nn.ReLU):
                    # Essayer de fusionner les deux couches
                    torch.quantization.fuse_modules(child, [str(i), str(i+1)], inplace=True)
                    i += 2
                    continue

                i += 1

            # Vérifier si l'enfant était en mode training
            if was_training:
                # Mettre l'enfant en mode train
                child.train()

    # IMPORTANT: retourner le même objet (les tests vérifient l'identité)
    # Retourner le modèle
    return model


class QATWrapper(nn.Module):
    """
    Wrapper pour QAT qui ajoute QuantStub et DeQuantStub.
    """
    def __init__(self, m):
        super().__init__()
        self.quant = aoq.QuantStub()
        self.m = m
        self.dequant = aoq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.m(x)
        x = self.dequant(x)
        return x


def prepare_qat(model, backend="qnnpack"):
    """
    Retourne une **copie prête pour QAT** de `model` :
      - Définit le backend quantifié (par défaut : 'qnnpack')
      - Applique une fusion optimale (Conv+BN(+Act))
      - Attache une qconfig QAT par défaut
      - Exécute prepare_qat en mode eager pour insérer des observateurs/fake-quant
      - Retourne le module préparé en mode **train()**

    Le modèle original `model` **ne doit pas** être muté.

    Parameters
    ----------
    model : nn.Module
        Modèle FP32 à préparer pour QAT.
    backend : str
        Moteur quantifié (utiliser 'fbgemm' sur x86 ; 'qnnpack' sur ARM).

    Returns
    -------
    nn.Module
        Un nouveau modèle prêt pour QAT (avec observateurs) en mode training.
    """
    # Configurer le backend quantifié avant toute opération
    has_quantized = hasattr(torch.backends, "quantized")
    has_engine = hasattr(torch.backends.quantized, "engine") if has_quantized else False
    if has_quantized and has_engine:
        try:
            torch.backends.quantized.engine = backend
        except Exception:
            pass

    # 1) Travailler sur une copie ; ne pas muter l'original
    # Créer une copie profonde du modèle et le mettre en mode train
    qat = copy.deepcopy(model).train()
    qat.eval()

    # 2) Fusionner les modules éligibles (optimale ; opération nulle sûre si non supporté)
    # Fusionner les modules éligibles (qat)
    fuse_model_inplace(qat)

    # 3) Attacher une qconfig QAT par défaut
    qat.qconfig = torch.quantization.get_default_qat_qconfig(backend)

    # 4) Préparer pour QAT (insérer observateurs/fake-quant)
    qat.train()
    # Préparer le modèle pour QAT
    torch.quantization.prepare_qat(
        qat,  # Le modèle à préparer pour QAT
        inplace=True,  # Définir la valeur correcte pour inplace
    )

    return qat


def train_qat_model(qat_model, train_loader, val_loader, num_epochs=1, device="cpu"):
    """
    Entraîne un modèle QAT avec fake-quantization.
    """
    # Définir la fonction de perte
    criterion = nn.CrossEntropyLoss()
    # Initialiser l'optimiseur
    optimizer = torch.optim.SGD(
        (p for p in qat_model.parameters() if p.requires_grad),
        lr=1e-4,
        momentum=0.9,
        weight_decay=1e-4
    )

    # Déplacer le modèle vers le périphérique
    qat_model.to(device)
    qat_model.train()

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Phase d'entraînement avec barre de progression
        pbar = tqdm(train_loader, desc=f"Époque {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = qat_model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistiques
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Mettre à jour la barre de progression
            current_acc = 100 * correct / total
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

        # Afficher les statistiques de l'époque
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Époque {epoch+1}/{num_epochs} terminée - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

    return qat_model


if __name__ == "__main__":
    # Dataloaders
    dm = ChestXRayDataModule(data_dir, batch_size=32)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Créer un QATResNet18 et charger les poids avec mapping
    model = helper_utils.resnet18_qat_ready_pretrained(num_classes=3, use_quant_stubs=False)
    model_weights = torch.load("../models/best_90_resnet18_chest_xray_classifier_weights_pruned.pth", map_location="cpu")
    model.load_state_dict(model_weights)
    # Envelopper le modèle de base dans QATWrapper pour ajouter les stubs de quantification
    wrapped_model = QATWrapper(model)

    # Calculer la précision du modèle chargé AVANT QAT
    base_accuracy = helper_utils.compute_accuracy(model, val_loader, DEVICE)
    print(f"Précision du modèle aavant QAT: {base_accuracy:.4f} ({base_accuracy*100:.2f}%)")

    wrapped_model = QATWrapper(model)

    print("Modèle de base chargé et enveloppé")

    # Préparer le modèle QAT
    qat_model = prepare_qat(wrapped_model, backend="qnnpack")
    print("Modèle préparé pour QAT")

    # Fine-tune avec fake-quant dans la boucle (QAT doit être sur CPU, pas MPS)
    # Learning rate légèrement plus élevé pour QAT (le modèle doit s'adapter à la quantification)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD((p for p in qat_model.parameters() if p.requires_grad),
                                lr=5e-4, momentum=0.9, weight_decay=1e-4)  # LR augmenté pour QAT

    # QAT nécessite CPU car les opérations fake-quant ne sont pas supportées sur MPS
    qat_model.to("cpu")
    cpu_device = torch.device("cpu")

    # QAT nécessite plusieurs époques pour s'adapter à la fake-quantization
    helper_utils.train_model_with_best_checkpoint_and_metrics(
        qat_model,
        train_loader,
        val_loader,
        1,
        optimizer,
        cpu_device,
        save_path="../models/quantized_int8_resnet18_chest_xray_weights.pth")

    qat_model.to("cpu")

    # Convertir en vrai INT8 (s'exécute sur CPU)
    qat_model.eval()
    int8_model = torch.quantization.convert(qat_model)
    print("Modèle converti en int8")

    # Sauvegarder le modèle quantifié avec l'état complet
    torch.save({
        'model_state_dict': int8_model.state_dict(),
        'quantization_config': int8_model.state_dict()
    }, "../models/quantized_int8_resnet18_chest_xray_weights.pth")

    print("Checkpoint du modèle quantifié sauvegardé dans quantized_int8_resnet18_chest_xray_weights.pth")

    # Évaluer le modèle int8 sur les données de validation
    int8_model.eval()
    print("Test du modèle sur cpu")
    test_acc = helper_utils.compute_accuracy(int8_model, val_loader, device="cpu")
    print(f"Précision du test dans le modèle de base: {base_accuracy:.2f}%")
    print(f'\nPrécision du test du modèle Int8: {test_acc:.2f}%')

    # Mesurer le temps d'inférence pour les deux modèles sur cpu
    model.to("cpu")
    int8_model.to("cpu")
    base_time = helper_utils.bench(model, device="cpu", shape=(32, 3, 224, 224))
    int8_time = helper_utils.bench(int8_model, device="cpu", shape=(32, 3, 224, 224))

    # Calculer le pourcentage d'amélioration
    time_improvement = ((base_time - int8_time) / base_time) * 100

    print(f"\nComparaison du temps d'inférence:")
    print(f"Modèle de base: {base_time:.4f} secondes par batch")
    print(f"Modèle Int8: {int8_time:.4f} secondes par batch")
    print(f"Amélioration de vitesse: {time_improvement:.1f}%")

    # Obtenir les tailles de fichier en MB
    base_size = os.path.getsize("../models/best_90_resnet18_chest_xray_classifier_weights_pruned.pth") / (1024 * 1024)
    int8_size = os.path.getsize("../models/quantized_int8_resnet18_chest_xray_weights.pth") / (1024 * 1024)

    print(f"\nComparaison de la taille des modèles:")
    print(f"Modèle de base: {base_size:.2f} MB")
    print(f"Modèle Int8: {int8_size:.2f} MB")
    print(f"Réduction de taille: {((base_size - int8_size) / base_size * 100):.1f}%")