import os
import time
import copy

import numpy as np
try:
    import seaborn as sns
except ImportError:
    sns = None
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision.models import resnet18
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    plt = None
    mticker = None

try:
    from torchvision.models import ResNet18_Weights
    TV_DEFAULT_WEIGHTS = ResNet18_Weights.DEFAULT
except Exception:
    TV_DEFAULT_WEIGHTS = None

try:
    from torchao.quantization import QuantStub, DeQuantStub
except Exception:
    try:
        from torch.ao.quantization import QuantStub, DeQuantStub
    except Exception:
        from torch.quantization import QuantStub, DeQuantStub


def bench(m, iters=20, shape = (16, 3, 224, 224), device="cpu"):
    torch.manual_seed(17)
    m.eval()
    x = torch.randn(shape).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = m(x)
    return (time.perf_counter() - start) / iters


def compute_accuracy(model, dataloader, device):
    """
    Calcule l'accuracy globale sur un dataloader.
    """
    # Mettre le modèle en mode évaluation
    model.eval()
    # Déplacer le modèle vers le périphérique
    model.to(device)
    # Initialiser les compteurs
    correct = 0
    total = 0
    # Désactiver les calculs de gradient pour l'inférence
    with torch.no_grad():
        # Parcourir les batches du dataloader
        for images, labels in dataloader:
            # Déplacer les images et labels vers le périphérique
            images = images.to(device)
            labels = labels.to(device)
            # Effectuer une passe forward
            outputs = model(images)
            # Obtenir les prédictions (classe avec la plus haute probabilité)
            preds = torch.argmax(outputs, dim=1)
            # Compter les prédictions correctes
            correct += (preds == labels).sum().item()
            # Compter le nombre total d'échantillons
            total += labels.numel()
    # Retourner l'accuracy (fraction de prédictions correctes)
    return (correct / total) if total else 0.0


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png", show=True):
    if plt is None or sns is None:
        print("matplotlib ou seaborn non disponible, impossible d'afficher la matrice de confusion")
        return save_path

    was_interactive = plt.isinteractive()
    if not show:
        plt.ioff()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Étiquettes prédites")
    plt.ylabel("Étiquettes réelles")
    plt.title("Matrice de confusion")

    if show:
        plt.show()
    else:
        plt.close()
        if was_interactive:
            plt.ion()

    return save_path


def per_class_acc_and_conf_matrix(trained_model, data_module):
    """
    Évalue un modèle entraîné sur un dataset de validation et affiche un rapport
    de précision par classe et une matrice de confusion.

    Args:
        trained_model: Le modèle Lightning entraîné à évaluer.
        data_module: Le module de données contenant le dataloader de validation.
    """
    # --- Configuration ---
    # Met le modèle en mode évaluation
    trained_model.eval()
    # Détermine le dispositif à utiliser pour le calcul
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Déplace le modèle vers le dispositif sélectionné
    trained_model = trained_model.to(device)

    # Initialise des listes pour stocker les prédictions et les étiquettes réelles
    all_preds = []
    all_labels = []

    # --- Exécution de l'inférence ---
    # Enveloppe le dataloader avec tqdm pour une barre de progression
    val_loader_with_progress = tqdm(
        data_module.val_dataloader(),
        # Définit une description pour la barre de progression
        desc="Évaluation du modèle",
        # Ne laisse pas la barre de progression après la fin
        leave=False
    )

    # Désactive les calculs de gradient pour l'inférence
    with torch.no_grad():
        # Parcourt les lots dans le dataloader de validation
        for batch in val_loader_with_progress:
            # Dépaquette les images et les étiquettes du lot
            images, labels = batch
            # Déplace les images et les étiquettes vers le dispositif approprié
            images, labels = images.to(device), labels.to(device)

            # Effectue une passe avant pour obtenir les sorties du modèle
            outputs = trained_model(images)
            # Obtient la classe prédite en trouvant l'index avec la valeur la plus élevée
            preds = torch.argmax(outputs, dim=1)

            # Ajoute les prédictions du lot actuel à la liste
            all_preds.append(preds)
            # Ajoute les étiquettes réelles du lot actuel à la liste
            all_labels.append(labels)

    # --- Calcul et affichage des métriques ---
    # Concatène toutes les prédictions en un seul tenseur
    all_preds = torch.cat(all_preds)
    # Concatène toutes les étiquettes réelles en un seul tenseur
    all_labels = torch.cat(all_labels)

    # Initialise la métrique de matrice de confusion
    num_classes = len(data_module.val_dataset.classes)
    confmat = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    # Calcule la matrice de confusion
    cm = confmat(all_preds, all_labels)

    # Calcule la précision par classe à partir de la matrice de confusion
    per_class_acc = cm.diag() / cm.sum(axis=1)
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    acc_global = (correct / total) if total else 0.0
    # Récupère les noms des classes depuis le dataset
    class_names = data_module.val_dataset.classes

    # Affiche un en-tête pour le rapport de précision
    print("--- Rapport de précision par classe ---")
    print(f"Précision globale : {acc_global:.4f}")
    # Parcourt chaque classe et affiche sa précision
    for i, acc in enumerate(per_class_acc):
        # Affiche la précision pour la classe actuelle
        print(f"  - Précision pour la classe '{class_names[i]}' : {acc.item():.4f}")
    # Affiche une nouvelle ligne pour l'espacement
    print()

    # Trace la matrice de confusion
    plot_confusion_matrix(cm.cpu().numpy(), class_names)


class QATBasicBlock(nn.Module):
    """
    Variante ResNet BasicBlock compatible QAT avec noms identiques à torchvision.
    Utilise FloatFunctional pour l'addition résiduelle (compatible quantification).
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # Noms identiques à torchvision ResNet18
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Chemin de downsample comme Sequential (ou None) pour compatibilité torchvision
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

        # Addition FloatFunctional pour que l'addition résiduelle soit consciente de la quantification
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.skip_add.add(out, identity)
        out = self.relu(out)
        return out


class QATResNet18(nn.Module):
    """
    Architecture ResNet18 compatible QAT avec noms identiques à torchvision.
    Inclut QuantStub/DeQuantStub pour que la préparation/conversion QAT fonctionne proprement.
    """
    def __init__(self, num_classes=1000, use_quant_stubs=False, hidden_size=512, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.use_quant_stubs = use_quant_stubs
        if use_quant_stubs:
            self.quant = QuantStub()
        else:
            self.quant = nn.Identity()

        # Noms identiques à torchvision ResNet18
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 étages : [2,2,2,2] blocs, strides : [1,2,2,2]
        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # FC Sequential correspondant aux poids sauvegardés
        num_ftrs = 512 * QATBasicBlock.expansion
        self.fc = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(num_ftrs, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden_size, num_classes)
        )

        if use_quant_stubs:
            self.dequant = DeQuantStub()
        else:
            self.dequant = nn.Identity()

    def _make_layer(self, inplanes, planes, blocks, stride):
        layers = [QATBasicBlock(inplanes, planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(QATBasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)
        return x



@torch.no_grad()
def load_imagenet_pretrained_into_qat_resnet18(model: nn.Module,
                                              weights=TV_DEFAULT_WEIGHTS,
                                              strict=False):
    """
    Charge les poids ResNet18 de torchvision dans QATResNet18.
    Les noms de couches sont maintenant identiques, seule la FC est ignorée.

    Returns:
        model (nn.Module): même instance avec les poids chargés.
        missing_keys (list[str]), unexpected_keys (list[str])
    """
    # Construire un resnet18 torchvision standard avec les poids ImageNet
    if TV_DEFAULT_WEIGHTS is not None:
        tv = resnet18(weights=weights)
    else:
        tv = resnet18(pretrained=True)  # torchvision plus ancien

    tv_sd = tv.state_dict()

    # Filtrer les clés FC (structure différente: Linear vs Sequential)
    new_sd = {k: v for k, v in tv_sd.items() if not k.startswith("fc.")}

    incompat = model.load_state_dict(new_sd, strict=strict)
    return model, list(incompat.missing_keys), list(incompat.unexpected_keys)


def resnet18_qat_ready_pretrained(num_classes=1000, use_quant_stubs=False,
                                   hidden_size=512, dropout1=0.3, dropout2=0.2):
    """
    Construit votre QATResNet18, charge les poids ImageNet et retourne le modèle FP32.
    Si num_classes != 1000, la couche FC est initialisée aléatoirement.
    """
    model_fp32 = QATResNet18(
        num_classes=num_classes,
        use_quant_stubs=use_quant_stubs,
        hidden_size=hidden_size,
        dropout1=dropout1,
        dropout2=dropout2
    )
    # Charger les poids ImageNet où les formes correspondent
    model_fp32, missing, unexpected = load_imagenet_pretrained_into_qat_resnet18(
        model_fp32, weights=TV_DEFAULT_WEIGHTS, strict=False
    )
    return model_fp32


def load_resnet18_from_checkpoint(weights_path, num_classes=None):
    """
    Charge un ResNet18 standard depuis un checkpoint.
    Gère les préfixes 'model.' et 'module.' automatiquement.
    """
    state = torch.load(weights_path, map_location="cpu")
    state = state.get("state_dict", state)

    state = {
        (k[6:] if k.startswith("model.") else k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
        if not k.startswith("loss_fn.")
    }

    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features

    if num_classes is not None:
        model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif "fc.weight" in state:
        num_classes = state["fc.weight"].shape[0]
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        hidden = state["fc.1.weight"].shape[0]
        num_classes = state["fc.4.weight"].shape[0]
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    model.load_state_dict(state, strict=True)
    return model


def train_model_with_best_checkpoint_and_metrics(
    model, train_loader, val_loader, num_epochs, optimizer, device, save_path=None
):
    """
    Entraîne un modèle et sauvegarde le meilleur checkpoint basé sur la précision de validation.

    Cette fonction exécute une boucle d'entraînement complète avec validation, en suivant
    la précision de validation et en sauvegardant le meilleur modèle. À la fin de l'entraînement,
    le modèle est restauré avec les poids du meilleur checkpoint.

    Args:
        model: Le modèle PyTorch à entraîner
        train_loader: DataLoader pour les données d'entraînement
        val_loader: DataLoader pour les données de validation
        num_epochs: Nombre d'époques d'entraînement
        optimizer: Optimiseur PyTorch (déjà configuré avec les paramètres du modèle)
        device: Périphérique sur lequel entraîner ('cpu', 'cuda', 'mps')
        save_path: Chemin optionnel pour sauvegarder le meilleur checkpoint

    Returns:
        model: Le modèle avec les poids du meilleur checkpoint chargés
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # Variables pour suivre le meilleur modèle
    best_val_acc = 0.0
    best_model_state = None

    # Boucle d'entraînement sur les époques
    for epoch in range(num_epochs):
        # --- Phase d'entraînement ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Époque {epoch+1}/{num_epochs} [Entraînement]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Passe forward, backward et mise à jour
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumuler les métriques d'entraînement
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Phase de validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc=f"Époque {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Accumuler les métriques de validation
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_acc = 100 * val_correct / val_total
                val_pbar.set_postfix(acc=f"{val_acc:.2f}%")

        # Calculer les métriques finales de l'époque
        val_acc = 100 * val_correct / val_total
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Époque {epoch+1}/{num_epochs} - Perte : {avg_train_loss:.4f} - Précision Val : {val_acc:.4f}")

        # Vérifier si c'est le meilleur modèle jusqu'à présent
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_model_state, save_path)
                print(f"Nouvelle meilleure précision : {best_val_acc:.4f}, modèle sauvegardé dans {save_path}")

    # Restaurer le meilleur modèle à la fin de l'entraînement
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nEntraînement terminé :\nMeilleure précision : {best_val_acc:.4f}\nPrécision finale : {val_acc:.4f}")
        if save_path:
            print(f"Modèle final sauvegardé dans {save_path}")

    return model


def training_loop_with_best_model(model, train_loader, val_loader, loss_fcn, optmzr, device, num_epochs=3, scheduler=None, model_name="best_model_nanobanana_pro.pth"):
    """
    Exécute la boucle d'entraînement et de validation pour un modèle PyTorch donné.
    Sauvegarde et retourne le modèle avec la meilleure précision de validation.

    Args:
        model: Le modèle à entraîner.
        train_loader: Le data loader pour le dataset d'entraînement.
        val_loader: Le data loader pour le dataset de validation.
        loss_fcn: La fonction de perte pour calculer la perte d'entraînement.
        optmzr: L'optimiseur pour mettre à jour les paramètres du modèle.
        device: Le périphérique (CPU ou CUDA) sur lequel le modèle et les données seront traités.
        num_epochs: Le nombre total d'époques pour l'entraînement.
        model_name: Le nom du fichier pour sauvegarder le meilleur modèle.

    Returns:
        L'objet modèle entraîné avec les poids qui ont atteint la meilleure précision de validation.
    """
    # Crée le répertoire pour sauvegarder le meilleur modèle s'il n'existe pas.
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, model_name)

    # Déplace le modèle vers le périphérique de calcul spécifié.
    model.to(device)
    # Assigne la fonction de perte fournie.
    loss_function = loss_fcn
    # Assigne l'optimiseur fourni.
    optimizer = optmzr

    # Détermine le nombre de classes depuis la couche de sortie finale du modèle.
    if hasattr(model, 'classifier'):
        num_classes = model.classifier[-1].out_features
    elif hasattr(model, 'fc'):
        num_classes = model.fc.out_features
    else:
        raise AttributeError("Modèle doit avoir 'classifier' ou 'fc' pour déterminer num_classes")

    # Initialise les métriques de précision, precision et recall pour la validation.
    val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='macro').to(device)
    val_precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    val_recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)

    # Initialise les variables pour suivre les meilleures performances.
    best_val_accuracy = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0

    # Liste pour stocker les métriques par époque
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Commence la boucle principale d'entraînement et de validation pour un nombre d'époques.
    for epoch in range(num_epochs):
        # Met le modèle en mode entraînement.
        model.train()
        # Initialise une variable pour accumuler la perte d'entraînement.
        running_loss = 0.0
        # Initialise un compteur pour les échantillons d'entraînement correctement classés.
        total_train_correct = 0
        # Initialise un compteur pour le total d'échantillons d'entraînement.
        total_train_samples = 0

        # Crée une barre de progression pour les batchs d'entraînement.
        train_progress_bar = tqdm(train_loader, desc=f"Époque {epoch + 1}/{num_epochs} Entraînement", unit="batch")
        # Itère sur les batchs depuis le data loader d'entraînement.
        for images, labels in train_progress_bar:
            # Déplace les images et labels vers le périphérique désigné.
            images, labels = images.to(device), labels.to(device)
            # Réinitialise les gradients de tous les tenseurs optimisés à zéro.
            optimizer.zero_grad()
            # Effectue une passe forward pour obtenir les sorties du modèle.
            outputs = model(images)
            # Calcule la perte entre les sorties et les vrais labels.
            loss = loss_function(outputs, labels)
            # Effectue la rétropropagation pour calculer les gradients.
            loss.backward()
            # Met à jour les poids du modèle en utilisant l'optimiseur.
            optimizer.step()

            # Accumule la perte et les compteurs d'échantillons.
            running_loss += loss.item() * labels.size(0)
            # Obtient la classe prédite avec la probabilité la plus élevée.
            _, predicted = torch.max(outputs, dim=1)
            # Met à jour le compteur pour les prédictions correctes.
            total_train_correct += (predicted == labels).sum().item()
            # Met à jour le compte total d'échantillons traités.
            total_train_samples += labels.size(0)

            # Calcule la perte moyenne pour l'époque actuelle.
            epoch_loss = running_loss / total_train_samples
            # Calcule la précision pour l'époque actuelle.
            epoch_acc = 100 * total_train_correct / total_train_samples
            # Met à jour la barre de progression avec la perte et la précision en temps réel.
            train_progress_bar.set_postfix(loss=f"{epoch_loss:.4f}", accuracy=f"{epoch_acc:.2f}%")

        # Commence la phase de validation.
        # Met le modèle en mode évaluation.
        model.eval()
        # Initialise un compteur pour le total d'échantillons de validation.
        total_val_samples = 0
        # Initialise une variable pour accumuler la perte de validation.
        val_loss = 0.0

        # Réinitialise les objets de métriques de validation pour la nouvelle époque.
        val_accuracy_metric.reset()
        val_precision_metric.reset()
        val_recall_metric.reset()

        # Désactive les calculs de gradient pour l'efficacité pendant la validation.
        with torch.no_grad():
            # Crée une barre de progression pour les batchs de validation.
            val_progress_bar = tqdm(val_loader, desc=f"Époque {epoch + 1}/{num_epochs} Validation", unit="batch")
            # Itère sur les batchs depuis le data loader de validation.
            for images, labels in val_progress_bar:
                # Déplace les images et labels vers le périphérique désigné.
                images, labels = images.to(device), labels.to(device)
                # Effectue une passe forward.
                outputs = model(images)
                # Calcule la perte.
                loss = loss_function(outputs, labels)
                # Accumule la perte de validation.
                val_loss += loss.item() * labels.size(0)
                # Obtient la classe prédite.
                _, predicted = torch.max(outputs, dim=1)
                # Met à jour le compte total d'échantillons de validation.
                total_val_samples += labels.size(0)

                # Met à jour les objets de métriques avec les prédictions et labels du batch.
                val_accuracy_metric.update(predicted, labels)
                val_precision_metric.update(predicted, labels)
                val_recall_metric.update(predicted, labels)

                # Met à jour la barre de progression avec la précision actuelle.
                val_progress_bar.set_postfix(
                    accuracy=f"{100 * val_accuracy_metric.compute():.2f}%"
                )

        # Calcule la perte de validation moyenne pour l'époque.
        avg_val_loss = val_loss / total_val_samples

        # Calcule la perte d'entraînement moyenne pour l'époque.
        avg_train_loss = running_loss / total_train_samples
        # Calcule la précision d'entraînement pour l'époque.
        train_acc = total_train_correct / total_train_samples

        # Calcule les valeurs de métriques finales pour l'époque entière.
        final_val_acc = val_accuracy_metric.compute()
        final_val_precision = val_precision_metric.compute()
        final_val_recall = val_recall_metric.compute()

        # Stocke les métriques pour cette époque
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(final_val_acc.item())

        # Affiche un résumé des résultats de validation pour l'époque.
        print(f'Perte Val (Moy): {avg_val_loss:.4f}, Précision Val: {final_val_acc * 100:.2f}%\n')

        # Vérifie si le modèle actuel est le meilleur et le sauvegarde.
        if final_val_acc > best_val_accuracy:
            best_val_accuracy = final_val_acc
            best_val_precision = final_val_precision
            best_val_recall = final_val_recall
            torch.save(model.state_dict(), best_model_path)
            print(f"Nouveau meilleur modèle sauvegardé dans {best_model_path} avec Précision Val: {best_val_accuracy * 100:.2f}%\n")

    # Affiche un message indiquant la fin de l'entraînement.
    print("\nEntraînement terminé. Meilleur modèle entraîné retourné.")
    print(f"Meilleure Exactitude Val: {best_val_accuracy * 100:.2f}%")
    print(f"Meilleure Precision Val: {best_val_precision:.4f}")
    print(f"Meilleur Recall Val: {best_val_recall:.4f}\n")

    # Charge les poids du meilleur modèle avant de le retourner.
    model.load_state_dict(torch.load(best_model_path))

    # Retourne le meilleur modèle entraîné et les métriques.
    metrics = (train_losses, train_accuracies, val_losses, val_accuracies)
    return model, metrics
