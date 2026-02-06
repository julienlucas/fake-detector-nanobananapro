import os
from dotenv import load_dotenv

# Charger le .env depuis le répertoire racine du projet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, ".env"))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
torch.set_float32_matmul_precision('medium')  # Utilise les Tensor Cores de la A10G

import io
import numpy as np
from PIL import Image, ImageEnhance
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar, StochasticWeightAveraging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchvision import models as tv_models
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from collections import Counter
from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfApi, login


# ============================================================
# CONFIGURATION: "Universal Real" - Modèle généraliste anti-fixette
# - Double Pooling (Avg + Max) pour détecter les artefacts
# - Focal Loss gamma=3.0 pour se concentrer sur les cas difficiles
# - Augmentations agressives pour détruire la signature du capteur
# - Weight Decay 1e-3 + Dropout 0.5 pour forcer la généralisation
# - Repartir du Trial #13 (base saine) plutôt que du modèle à 92% (mémorisé)
# ============================================================
CONFIG = {
    # Checkpoint du trial #13 sur HuggingFace
    "hf_repo": "julienlucas/fakefinder",
    "hf_filename": "models/trial_13_efficientnetv2s_0.9080.pth",

    # Dataset et output sur HuggingFace
    "hf_dataset_repo": "julienlucas/midjourney-dalle-sd-nanobananapro-dataset",
    "hf_output_repo": "julienlucas/fakefinder",

    # Hyperparamètres gagnants du trial #13
    "learning_rate": 2.54e-4,
    "lr_ratio": 0.05,
    "weight_decay": 1e-3,  # Augmenté pour forcer le modèle à utiliser tous ses neurones (anti-fixette)
    "dropout": 0.5,  # Augmenté à 50% pour forcer la généralisation (anti-mémorisation)
    "label_smoothing": 0.0,
    "focal_gamma": 3.0,  # Élevé pour forcer le modèle à se concentrer sur les cas difficiles (fakes subtils) et améliorer le F1

    # Entraînement long pour aller au-delà de 90%
    "batch_size": 16,
    "accumulate_grad": 4,  # Batch effectif = 64
    "max_epochs": 25,
    "patience": 6,

    # Dégel (trial #13: blocs=4, epoch=2)
    "unfreeze_blocks": 4,
    "unfreeze_epoch": 2,

    "image_size": 384,
}


# ============================================================
# AUGMENTATIONS DE TEXTURE
# ============================================================

class RandomJPEGCompression:
    """Simule la compression JPEG des réseaux sociaux."""
    def __init__(self, quality_min=30, quality_max=95, p=0.3):
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
        return Image.open(buffer).convert('RGB')


class RandomSharpness:
    """Variation de netteté pour détecter les textures."""
    def __init__(self, factor_range=(0.5, 2.5), p=0.2):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img
        factor = np.random.uniform(*self.factor_range)
        return ImageEnhance.Sharpness(img).enhance(factor)


class RandomGaussianNoise:
    """Bruit gaussien léger (sensor noise). Appliqué sur tenseur après ToTensor. Bruit + Selfie = Real."""
    def __init__(self, std=0.02, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        noise = torch.randn_like(x) * self.std
        return (x + noise).clamp(0, 1)


class HFDatasetWrapper(Dataset):
    """Wrapper pour convertir un dataset Hugging Face en Dataset PyTorch avec transforms"""
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.classes = hf_dataset.features['label'].names
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self._labels = [item['label'] for item in hf_dataset]

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['label']

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class FakeDetectorDataModule(pl.LightningDataModule):
    """DataModule pour fake/real detection - charge depuis Hugging Face"""

    def __init__(self, hf_dataset_repo, batch_size=16, image_size=384, token=None):
        super().__init__()
        self.hf_dataset_repo = hf_dataset_repo
        self.batch_size = batch_size
        self.token = token

        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),

            # 1. Détruire la signature du capteur (JPEG agressif aléatoire)
            RandomJPEGCompression(quality_min=20, quality_max=80, p=0.6),

            # 2. Simuler les différents focus/optiques (Flou vs Netteté)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            ], p=0.4),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.4),

            # 3. Simuler les HDR/Smartphones (Variations de lumière)
            transforms.RandomAutocontrast(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),

            # 4. Augmentations géométriques (RandomPerspective = distorsion selfie XR / focale 85mm IA)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),

            transforms.ToTensor(),
            RandomGaussianNoise(std=0.02, p=0.5),  # Bruit capteur léger : Bruit + Selfie = Real
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.class_weights = None

    def setup(self, stage=None):
        print(f"Chargement {self.hf_dataset_repo} depuis Hugging Face...")
        ds = load_dataset(self.hf_dataset_repo, token=self.token)

        self.train_dataset = HFDatasetWrapper(ds['train'], self.train_transform)
        self.val_dataset = HFDatasetWrapper(ds['test'], self.val_transform)

        all_labels = [item['label'] for item in ds['train']]

        class_counts = Counter(all_labels)
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        self.class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]

        print(f"Classes: {self.train_dataset.classes}")
        print(f"Class weights: {self.class_weights}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        all_labels = self.train_dataset._labels
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


class HFBestModelCallback(pl.Callback):
    """
    À chaque fin d'epoch de validation, si val_f1 est meilleur que le record,
    sauvegarde le modèle en .pth et l'upload sur HuggingFace.
    On utilise val_f1 (et non val_acc) pour éviter le biais vers une seule classe.
    """
    def __init__(self, hf_api, repo_id, token):
        self.hf_api = hf_api
        self.repo_id = repo_id
        self.token = token
        self.best_f1 = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        val_f1 = trainer.callback_metrics.get('val_f1')
        val_acc = trainer.callback_metrics.get('val_acc')
        if val_f1 is None:
            return

        f1 = val_f1.item()
        if f1 <= self.best_f1:
            return

        self.best_f1 = f1
        acc = val_acc.item() if val_acc is not None else 0.0
        acc_str = f"{acc*100:.1f}"
        f1_str = f"{f1*100:.1f}"

        # Sauvegarde locale en .pth
        os.makedirs("../models", exist_ok=True)
        local_path = f"../models/best_run2_f1{f1_str}_acc{acc_str}_efficientnetv2s_fake_detector.pth"
        torch.save({
            'model_state_dict': pl_module.backbone.state_dict(),
            'classifier_state_dict': pl_module.head.state_dict(),
            'pool_avg': pl_module.pool_avg.state_dict(),
            'pool_max': pl_module.pool_max.state_dict(),
            'num_ftrs': pl_module.num_ftrs,
            'config': pl_module.config,
            'accuracy': acc,
            'f1_score': f1
        }, local_path)

        # Upload sur HuggingFace
        hf_path = f"models/best_f1{f1_str}_acc{acc_str}_efficientnetv2s_fake_detector.pth"
        try:
            self.hf_api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hf_path,
                repo_id=self.repo_id,
                repo_type="model",
                token=self.token
            )
            print(f"\n>>> Epoch {trainer.current_epoch}: Nouveau best F1={f1_str}% (acc={acc_str}%) uploade -> {self.repo_id}/{hf_path}\n")
        except Exception as e:
            print(f"\n>>> Erreur upload HF: {e}\n")


class PhaseChangeCallback(pl.Callback):
    """Dégèle les N derniers blocs du backbone à l'époque prévue."""
    def __init__(self, config):
        self.config = config
        self.unfreeze_done = False

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.config["unfreeze_epoch"] and not self.unfreeze_done:
            total_blocks = len(pl_module.backbone.features)
            blocks_to_unfreeze = range(
                total_blocks - self.config["unfreeze_blocks"],
                total_blocks
            )

            unfrozen = 0
            for idx in blocks_to_unfreeze:
                for param in pl_module.backbone.features[idx].parameters():
                    param.requires_grad = True
                    unfrozen += 1

            trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in pl_module.parameters())

            print(f"\n>>> EPOCH {trainer.current_epoch}: {self.config['unfreeze_blocks']} DERNIERS BLOCS DEGELES ({unfrozen} params)")
            print(f"Parametres entrainables: {trainable:,} / {total:,}\n")
            self.unfreeze_done = True


class FakeDetectorFineTuner(pl.LightningModule):
    """
    Fine-tuning du meilleur modèle Optuna (trial #13, 90.5%)
    avec BatchNorm et entraînement long pour viser 95%.
    """
    def __init__(
        self,
        num_classes=2,
        class_weights=None,
        pretrained_checkpoint=None,
        config=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_checkpoint'])

        self.config = config or CONFIG
        self.num_classes = num_classes

        # ============================================================
        # 1. BACKBONE: torchvision EfficientNet-V2-S
        # ============================================================
        model = tv_models.efficientnet_v2_s(weights=None)
        num_ftrs = model.classifier[1].in_features
        self.num_ftrs = num_ftrs
        model.classifier = nn.Identity()

        # ============================================================
        # 2. CHARGER LE BACKBONE DU TRIAL #13
        # ============================================================
        if pretrained_checkpoint is not None:
            backbone_state = pretrained_checkpoint['model_state_dict']
            model.load_state_dict(backbone_state, strict=False)
            trial_acc = pretrained_checkpoint.get('accuracy', 'N/A')
            print(f"Backbone du trial #13 charge (accuracy: {trial_acc})")
        else:
            imagenet_model = tv_models.efficientnet_v2_s(weights=tv_models.EfficientNet_V2_S_Weights.DEFAULT)
            backbone_state = {k: v for k, v in imagenet_model.state_dict().items()
                             if not k.startswith('classifier')}
            model.load_state_dict(backbone_state, strict=False)
            print("Backbone ImageNet charge (pas de checkpoint)")

        self.backbone = model
        # Double Pooling (Avg + Max) pour améliorer la détection des fakes
        # Avg Pooling: moyenne des features (détecte les patterns généraux)
        # Max Pooling: valeurs maximales (détecte les artefacts/imperfections)
        # Concat des deux permet de capturer à la fois les patterns et les anomalies
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs * 2, 512),  # 2x car Avg + Max concaténés
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=self.config["dropout"]),
            nn.Linear(512, num_classes)
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.head.parameters())
        total = sum(p.numel() for p in self.backbone.parameters()) + trainable
        print(f"Phase 1: {trainable:,} / {total:,} parametres entrainables (head + BatchNorm)")

        # ============================================================
        # 6. LOSS & METRICS
        # ============================================================
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
        features = self.backbone.features(x)
        # Double pooling: Avg (patterns généraux) + Max (artefacts/anomalies)
        avg_p = self.pool_avg(features)  # Moyenne des features
        max_p = self.pool_max(features)  # Valeurs maximales (détecte les pixels "trop parfaits" ou "trop bruités")
        x = torch.cat([avg_p, max_p], dim=1)  # Concat pour combiner les deux types d'information
        return self.head(x)

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

        self.val_acc.update(logits, y)
        self.val_f1.update(logits, y)
        self.val_precision.update(logits, y)
        self.val_recall.update(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()

        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_prec_fake', precision[0])
        self.log('val_prec_real', precision[1])
        self.log('val_recall_fake', recall[0])
        self.log('val_recall_real', recall[1])

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        # IMPORTANT: Inclure le backbone même s'il est gelé (requires_grad=False)
        # Cela permet à l'optimizer de le "voir" et de le dégeler proprement plus tard
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config["learning_rate"] * self.config["lr_ratio"]},
            {'params': head_params, 'lr': self.config["learning_rate"]}
        ], weight_decay=self.config["weight_decay"])

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def download_checkpoint(token=None):
    """Télécharge le checkpoint du trial #13 depuis HuggingFace."""
    print(f"Telechargement du checkpoint depuis {CONFIG['hf_repo']}...")
    path = hf_hub_download(
        repo_id=CONFIG["hf_repo"],
        filename=CONFIG["hf_filename"],
        repo_type="model",
        token=token
    )
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    print(f"Checkpoint telecharge: {path}")
    if 'hyperparams' in checkpoint:
        print(f"Hyperparams du trial: {checkpoint['hyperparams']}")
    return checkpoint


def main():
    pl.seed_everything(42)

    # Connexion HuggingFace
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_ACCESS_TOKEN")
    hf_api = None
    if token:
        token = token.strip() if isinstance(token, str) else token
        login(token=token)
        hf_api = HfApi(token=token)
        print(f"Connexion HuggingFace etablie - Upload vers {CONFIG['hf_output_repo']}")
    else:
        print("Token HF non trouve - Upload desactive")

    # Télécharger le checkpoint du trial #13
    checkpoint = download_checkpoint(token=token)

    lr_backbone = CONFIG["learning_rate"] * CONFIG["lr_ratio"]
    print("\n" + "="*60)
    print("FINE-TUNING 'UNIVERSAL REAL' - ANTI-FIXETTE")
    print("="*60)
    print(f"Dataset: {CONFIG['hf_dataset_repo']} (HuggingFace)")
    print(f"Backbone: Trial #13 EfficientNet-V2-S (base saine)")
    print(f"Strategie 'Universal Real':")
    print(f"  - Double Pooling (Avg + Max) pour détecter les artefacts")
    print(f"  - Focal Loss gamma={CONFIG['focal_gamma']} pour les cas difficiles")
    print(f"  - Augmentations agressives (JPEG p=0.6, Blur p=0.4, Sharpness p=0.4)")
    print(f"  - Weight Decay={CONFIG['weight_decay']} pour forcer l'utilisation de tous les neurones")
    print(f"  - Dropout={CONFIG['dropout']} pour forcer la généralisation (anti-mémorisation)")
    print(f"LR head: {CONFIG['learning_rate']:.6f}")
    print(f"LR backbone: {lr_backbone:.6f}")
    print(f"Weight decay: {CONFIG['weight_decay']:.6f}")
    print(f"Dropout: {CONFIG['dropout']}")
    print(f"Label smoothing: {CONFIG['label_smoothing']}")
    print(f"Scheduler: cosine_warmup (10%)")
    print(f"Epochs: {CONFIG['max_epochs']} (patience={CONFIG['patience']})")
    print(f"Batch effectif: {CONFIG['batch_size']} x {CONFIG['accumulate_grad']} = {CONFIG['batch_size'] * CONFIG['accumulate_grad']}")
    print(f"Degel: {CONFIG['unfreeze_blocks']} blocs a epoch {CONFIG['unfreeze_epoch']}")
    print("="*60 + "\n")

    dm = FakeDetectorDataModule(
        hf_dataset_repo=CONFIG["hf_dataset_repo"],
        batch_size=CONFIG["batch_size"],
        image_size=CONFIG["image_size"],
        token=token
    )
    dm.setup()

    model = FakeDetectorFineTuner(
        num_classes=2,
        class_weights=dm.class_weights,
        pretrained_checkpoint=checkpoint,
        config=CONFIG
    )

    early_stop = EarlyStopping(
        monitor="val_f1",
        patience=CONFIG["patience"],
        mode="max",
        min_delta=0.001,
        verbose=True
    )

    progress = TQDMProgressBar(refresh_rate=1)
    phase_callback = PhaseChangeCallback(CONFIG)

    # Callback pour upload le meilleur .pth sur HuggingFace à chaque amélioration
    hf_upload_cb = None
    if hf_api is not None:
        hf_upload_cb = HFBestModelCallback(hf_api, CONFIG["hf_output_repo"], token)

    swa = StochasticWeightAveraging(
        swa_lrs=5e-5,
        swa_epoch_start=CONFIG["max_epochs"] - 7
    )

    # Détection du matériel
    if torch.cuda.is_available():
        accelerator = "gpu"
        if torch.cuda.get_device_capability()[0] >= 8:
            precision = "bf16-mixed"
            print("GPU CUDA Ampere+ - bf16-mixed")
        else:
            precision = "16-mixed"
            print("GPU CUDA - 16-mixed")
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        precision = "32"
        print("GPU MPS (Apple Silicon)")
    else:
        accelerator = "cpu"
        precision = "32"
        print("CPU uniquement")

    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=[cb for cb in [early_stop, progress, phase_callback, swa, hf_upload_cb] if cb is not None],
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=CONFIG["accumulate_grad"],
        log_every_n_steps=20
    )

    print("\n" + "="*60)
    print("DEMARRAGE DU FINE-TUNING")
    print("="*60 + "\n")

    trainer.fit(model, dm)

    print("\n" + "="*60)
    print("ENTRAINEMENT TERMINE")
    print("="*60)

    if hf_upload_cb is not None:
        print(f"\nMeilleur F1: {hf_upload_cb.best_f1*100:.2f}%")
    else:
        val_f1 = trainer.callback_metrics.get('val_f1')
        val_acc = trainer.callback_metrics.get('val_acc')
        if val_f1 is not None:
            print(f"\nMeilleur F1: {val_f1.item()*100:.2f}%")
        if val_acc is not None:
            print(f"Accuracy: {val_acc.item()*100:.2f}%")


if __name__ == "__main__":
    main()
