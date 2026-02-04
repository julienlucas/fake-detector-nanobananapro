"""
Fine-tuning du modèle ResNet18 FakeDetector à partir des poids Optuna trial 8 (86.3%)
Objectif: Atteindre 88-90% de précision
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchvision import models as tv_models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Poids pré-entraînés (trial 8 Optuna - 86.3%) - fichier local
    "pretrained_path": "./models/best_87_resnet18_fake_detector.pth",

    # Fine-tuning
    "learning_rate": 3e-5,          # Très bas pour ne pas détruire les features
    "lr_backbone_ratio": 0.1,       # Backbone 10x plus lent que la tête
    "weight_decay": 1e-5,
    "batch_size": 32,
    "max_epochs": 15,
    "patience": 6,                  # Early stopping

    # Architecture de la tête (DOIT correspondre au modèle Optuna)
    "hidden_size1": 512,
    "hidden_size2": 256,
    "dropout1": 0.33,               # Valeurs du trial 8
    "dropout2": 0.29,

    # Dégel progressif
    "unfreeze_epoch": 3,            # Dégeler backbone après N epochs
    "unfreeze_layers": ["layer4", "layer3"],

    # Focal Loss (comme Optuna)
    "focal_gamma": 1.85,
    "label_smoothing": 0.05,
}


class ConcatDataset(torch.utils.data.Dataset):
    """Combine plusieurs datasets ImageFolder"""
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = [0]
        for length in self.lengths:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
        self.classes = datasets[0].classes
        self.class_to_idx = datasets[0].class_to_idx

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths[1:], 1):
            if idx < cum_len:
                dataset_idx = i - 1
                break
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]


class FakeDetectorDataModule(pl.LightningDataModule):
    """DataModule pour le dataset local fake/real"""

    def __init__(self, data_base_dir, batch_size=32, aug_strength=0.5):
        super().__init__()
        self.data_base_dir = data_base_dir
        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2 * aug_strength,
                contrast=0.2 * aug_strength,
                saturation=0.2 * aug_strength,
                hue=0.05 * aug_strength
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            ], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.class_weights = None

    def setup(self, stage=None):
        midjourney_train = os.path.join(self.data_base_dir, "AIvsReal_midjourney_dalle_sd", "train")
        nanobanana_train = os.path.join(self.data_base_dir, "AIvsReal_nanobanana_pro", "train")

        midjourney_train_ds = datasets.ImageFolder(midjourney_train, self.train_transform)
        nanobanana_train_ds = datasets.ImageFolder(nanobanana_train, self.train_transform)
        self.train_dataset = ConcatDataset(midjourney_train_ds, nanobanana_train_ds)

        midjourney_test = os.path.join(self.data_base_dir, "AIvsReal_midjourney_dalle_sd", "test")
        nanobanana_test = os.path.join(self.data_base_dir, "AIvsReal_nanobanana_pro", "test")
        midjourney_val_ds = datasets.ImageFolder(midjourney_test, self.val_transform)
        nanobanana_val_ds = datasets.ImageFolder(nanobanana_test, self.val_transform)
        self.val_dataset = ConcatDataset(midjourney_val_ds, nanobanana_val_ds)

        # Calculer les poids de classes
        all_labels = []
        for dataset in [midjourney_train_ds, nanobanana_train_ds]:
            for _, label in dataset.samples:
                all_labels.append(label)

        class_counts = Counter(all_labels)
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        self.class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]

        print(f"Classes: {self.train_dataset.classes}")
        print(f"Class weights: {self.class_weights}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        all_labels = []
        for dataset in self.train_dataset.datasets:
            for _, label in dataset.samples:
                all_labels.append(label)
        weights = [self.class_weights[label] for label in all_labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )


class FocalLoss(nn.Module):
    """Focal Loss pour mieux gérer les cas difficiles"""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
        else:
            alpha = None
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class FakeDetectorFineTuner(pl.LightningModule):
    """
    Fine-tuner pour ResNet18 FakeDetector
    Charge les poids Optuna et continue l'entraînement
    """

    def __init__(
        self,
        num_classes=2,
        class_weights=None,
        pretrained_state_dict=None,
        config=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_state_dict'])

        self.config = config or CONFIG
        self.unfreeze_done = False

        # Créer le modèle avec la MÊME architecture que Optuna
        model = tv_models.resnet18(weights=None)
        num_ftrs = model.fc.in_features  # 512

        # Architecture de la tête (identique à Optuna)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, self.config["hidden_size1"]),
            nn.BatchNorm1d(self.config["hidden_size1"]),
            nn.ReLU(),
            nn.Dropout(self.config["dropout1"]),
            nn.Linear(self.config["hidden_size1"], self.config["hidden_size2"]),
            nn.BatchNorm1d(self.config["hidden_size2"]),
            nn.ReLU(),
            nn.Dropout(self.config["dropout2"]),
            nn.Linear(self.config["hidden_size2"], num_classes)
        )

        # Charger les poids pré-entraînés
        if pretrained_state_dict is not None:
            model.load_state_dict(pretrained_state_dict, strict=True)
            print("Poids Optuna chargés avec succès (architecture complète)")

        # Geler le backbone, garder la tête entraînable
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Paramètres entraînables: {trainable:,} / {total:,}")

        self.model = model

        # Loss function
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.loss_fn = FocalLoss(
            alpha=class_weights_tensor,
            gamma=self.config["focal_gamma"],
            label_smoothing=self.config["label_smoothing"]
        )

        # Métriques
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average=None)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.train_acc(logits, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_start(self):
        # Dégel progressif du backbone
        if self.current_epoch == self.config["unfreeze_epoch"] and not self.unfreeze_done:
            unfrozen = 0
            for name, param in self.model.named_parameters():
                for layer in self.config["unfreeze_layers"]:
                    if layer in name:
                        param.requires_grad = True
                        unfrozen += 1

            print(f"\n{'='*50}")
            print(f"Epoch {self.current_epoch}: Dégel de {self.config['unfreeze_layers']}")
            print(f"Paramètres dégelés: {unfrozen}")
            print(f"{'='*50}\n")
            self.unfreeze_done = True

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = self.val_acc(logits, y)
        f1 = self.val_f1(logits, y)
        precision = self.val_precision(logits, y)
        recall = self.val_recall(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)

        # Métriques par classe
        self.log('val_prec_fake', precision[0], on_step=False, on_epoch=True)
        self.log('val_prec_real', precision[1], on_step=False, on_epoch=True)
        self.log('val_recall_fake', recall[0], on_step=False, on_epoch=True)
        self.log('val_recall_real', recall[1], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Groupes de paramètres avec LR différents
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config["learning_rate"] * self.config["lr_backbone_ratio"]},
            {'params': head_params, 'lr': self.config["learning_rate"]}
        ], weight_decay=self.config["weight_decay"])

        # Cosine annealing avec warmup
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.01, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


def load_optuna_weights(path):
    """Charge les poids du modèle Optuna depuis un fichier local"""
    print(f"Chargement des poids depuis {path}...")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de poids introuvable: {path}")

    checkpoint = torch.load(path, map_location='cpu', weights_only=True)

    # Le fichier Optuna contient: model_state_dict, trial_number, accuracy, hyperparams
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        accuracy = checkpoint.get('accuracy', 'N/A')
        hyperparams = checkpoint.get('hyperparams', {})
        print(f"Poids chargés - Accuracy originale: {accuracy}")
        if hyperparams:
            print(f"Hyperparams: dropout1={hyperparams.get('dropout1', 'N/A'):.3f}, dropout2={hyperparams.get('dropout2', 'N/A'):.3f}")
    else:
        state_dict = checkpoint
        print("Poids chargés (format direct)")

    return state_dict


def main():
    pl.seed_everything(42)

    # Charger les poids Optuna depuis fichier local
    pretrained_state_dict = load_optuna_weights(CONFIG["pretrained_path"])

    # Dataset local
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_base_dir = os.path.join(base_dir, "..", "fake-detector-nanobananapro")

    print(f"\nDataset: {data_base_dir}")
    print(f"Config: LR={CONFIG['learning_rate']}, epochs={CONFIG['max_epochs']}")
    print(f"Dégel à epoch {CONFIG['unfreeze_epoch']}: {CONFIG['unfreeze_layers']}\n")

    # DataModule
    dm = FakeDetectorDataModule(
        data_base_dir,
        batch_size=CONFIG["batch_size"],
        aug_strength=0.5
    )
    dm.setup()

    # Modèle
    model = FakeDetectorFineTuner(
        num_classes=2,
        class_weights=dm.class_weights,
        pretrained_state_dict=pretrained_state_dict,
        config=CONFIG
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_acc",
        patience=CONFIG["patience"],
        mode="max",
        min_delta=0.002,
        verbose=True
    )

    checkpoint = ModelCheckpoint(
        dirpath="./models",
        filename="finetuned_{epoch:02d}_{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        verbose=True
    )

    progress = TQDMProgressBar(refresh_rate=1)

    # Accelerator
    if torch.cuda.is_available():
        accelerator = "gpu"
        precision = "16-mixed"
        print("GPU CUDA détecté")
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        precision = "32"
        print("GPU MPS (Apple Silicon) détecté")
    else:
        accelerator = "cpu"
        precision = "32"
        print("CPU uniquement")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=[early_stop, checkpoint, progress],
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        log_every_n_steps=20
    )

    # Entraînement
    print("\n" + "="*60)
    print("DÉMARRAGE DU FINE-TUNING")
    print("Objectif: 86.3% → 88-90%")
    print("="*60 + "\n")

    trainer.fit(model, dm)

    # Sauvegarder le meilleur modèle
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*60)

    best_acc = checkpoint.best_model_score
    if best_acc is not None:
        print(f"\nMeilleure accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"Checkpoint: {checkpoint.best_model_path}")

    # Sauvegarder aussi en format simple
    os.makedirs("./models", exist_ok=True)
    final_path = "./models/best_finetuned_resnet18_fake_detector.pth"
    torch.save(model.model.state_dict(), final_path)
    print(f"Modèle final sauvegardé: {final_path}")


if __name__ == "__main__":
    main()
