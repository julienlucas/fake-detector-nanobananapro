import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import io
import numpy as np
from PIL import Image
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
    "pretrained_path": "../models/best_82.8_mobilenetv3_fake_detector.pth",

    "learning_rate_head_phase1": 1e-3,
    "learning_rate_head_phase2": 1e-4,
    "learning_rate_backbone": 1e-5,
    "weight_decay": 5e-4,
    "batch_size": 32,
    "max_epochs": 20,
    "patience": 8,

    "dropout1": 0.3,

    "phase1_epochs": 5,
    "unfreeze_epoch": 6,

    "focal_gamma": 2.0,
    "label_smoothing": 0.1,
    "image_size": 256,
}


class GaussianNoise:
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise

    def __repr__(self):
        return f"GaussianNoise(std={self.std})"


class RandomJPEGCompression:
    def __init__(self, quality_min=30, quality_max=90, p=0.5):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img
        quality = np.random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def __repr__(self):
        return f"RandomJPEGCompression(quality_range=({self.quality_min}, {self.quality_max}), p={self.p})"


class ConcatDataset(torch.utils.data.Dataset):
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
    def __init__(self, data_base_dir, batch_size=32, aug_strength=0.5):
        super().__init__()
        self.data_base_dir = data_base_dir
        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(256, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            RandomJPEGCompression(quality_min=30, quality_max=90, p=0.15),
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
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.15))
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
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


class PhaseChangeCallback(pl.Callback):
    def __init__(self, config):
        self.config = config
        self.unfreeze_done = False

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.config["unfreeze_epoch"] and not self.unfreeze_done:
            for param in pl_module.model.parameters():
                param.requires_grad = True

            trainable = sum(p.numel() for p in pl_module.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in pl_module.model.parameters())

            optimizer = trainer.optimizers[0]
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0:
                    param_group['lr'] = self.config["learning_rate_backbone"]
                else:
                    param_group['lr'] = self.config["learning_rate_head_phase2"]

            print(f"\n>>> EPOCH {trainer.current_epoch}: BACKBONE DÉGELÉ")
            print(f"Paramètres entraînables: {trainable:,} / {total:,}")
            print(f"LR backbone: {self.config['learning_rate_backbone']}")
            print(f"LR tête: {self.config['learning_rate_head_phase2']}\n")
            self.unfreeze_done = True


class FakeDetectorFineTuner(pl.LightningModule):
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

        model = tv_models.mobilenet_v3_large(weights=None)

        num_ftrs = model.classifier[0].in_features

        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=self.config["dropout1"]),
            nn.Linear(1280, num_classes)
        )

        if pretrained_state_dict is not None:
            model.load_state_dict(pretrained_state_dict, strict=True)
            print("Poids pré-entraînés chargés avec succès")

        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Phase 1: Paramètres entraînables: {trainable:,} / {total:,} (classifier uniquement)")

        self.model = model

        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.loss_fn = FocalLoss(
            alpha=class_weights_tensor,
            gamma=self.config["focal_gamma"],
            label_smoothing=self.config["label_smoothing"]
        )

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

        self.log('val_prec_fake', precision[0], on_step=False, on_epoch=True)
        self.log('val_prec_real', precision[1], on_step=False, on_epoch=True)
        self.log('val_recall_fake', recall[0], on_step=False, on_epoch=True)
        self.log('val_recall_real', recall[1], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        backbone_params = [p for n, p in self.model.named_parameters() if 'classifier' not in n]
        head_params = [p for n, p in self.model.named_parameters() if 'classifier' in n]

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 0.0},
            {'params': head_params, 'lr': self.config["learning_rate_head_phase1"]}
        ], weight_decay=self.config["weight_decay"])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["max_epochs"],
            eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }



def load_pretrained_weights(path):
    print(f"Chargement des poids depuis {path}...")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de poids introuvable: {path}")

    checkpoint = torch.load(path, map_location='cpu', weights_only=True)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        accuracy = checkpoint.get('accuracy', 'N/A')
        print(f"Poids chargés - Accuracy originale: {accuracy}")
    else:
        state_dict = checkpoint
        print("Poids chargés (format direct)")

    return state_dict


def main():
    pl.seed_everything(42)

    pretrained_state_dict = load_pretrained_weights(CONFIG["pretrained_path"])

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_base_dir = os.path.join(base_dir, "..", "fake-detector-nanobananapro")

    print(f"\nDataset: {data_base_dir}")
    print(f"Config: Phase 1 (epochs 1-{CONFIG['phase1_epochs']}): LR_head={CONFIG['learning_rate_head_phase1']}")
    print(f"Config: Phase 2 (epochs {CONFIG['unfreeze_epoch']}-{CONFIG['max_epochs']}): LR_backbone={CONFIG['learning_rate_backbone']}, LR_head={CONFIG['learning_rate_head_phase2']}")
    print(f"Résolution: {CONFIG['image_size']}x{CONFIG['image_size']}, Focal gamma: {CONFIG['focal_gamma']}\n")

    dm = FakeDetectorDataModule(
        data_base_dir,
        batch_size=CONFIG["batch_size"],
        aug_strength=0.5
    )
    dm.setup()

    model = FakeDetectorFineTuner(
        num_classes=2,
        class_weights=dm.class_weights,
        pretrained_state_dict=pretrained_state_dict,
        config=CONFIG
    )

    early_stop = EarlyStopping(
        monitor="val_acc",
        patience=CONFIG["patience"],
        mode="max",
        min_delta=0.002,
        verbose=True
    )

    checkpoint = ModelCheckpoint(
        dirpath="../models",
        filename="finetuned_mobilenetv3_{epoch:02d}_{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        verbose=True
    )

    progress = TQDMProgressBar(refresh_rate=1)
    phase_callback = PhaseChangeCallback(CONFIG)

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

    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=[early_stop, checkpoint, progress, phase_callback],
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        log_every_n_steps=20
    )

    print("\n" + "="*60)
    print("DÉMARRAGE DU FINE-TUNING MobileNet V3 Large")
    print("="*60 + "\n")

    trainer.fit(model, dm)

    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*60)

    best_acc = checkpoint.best_model_score
    if best_acc is not None:
        print(f"\nMeilleure accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"Checkpoint: {checkpoint.best_model_path}")

    os.makedirs("../models", exist_ok=True)
    final_path = "../models/best_finetuned_mobilenetv3_fake_detector.pth"
    torch.save(model.model.state_dict(), final_path)
    print(f"Modèle final sauvegardé: {final_path}")


if __name__ == "__main__":
    main()
