import os
from dotenv import load_dotenv

# Charger le .env depuis le répertoire racine du projet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, ".env"))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar, Callback
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchvision import models as tv_models
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from collections import Counter
from datasets import load_dataset
from huggingface_hub import HfApi, login, hf_hub_download
from PIL import Image
import pandas as pd
import copy

# Global pour stocker les meilleurs modèles (top 5)
BEST_MODELS = []  # Liste de (score, trial_number, model_state_dict, hyperparams)
MAX_BEST_MODELS = 5

# Global pour l'API HuggingFace et le repo
HF_API = None
HF_OUTPUT_REPO = None
HF_TOKEN = None


class HFDatasetWrapper(Dataset):
    """Wrapper pour convertir un dataset Hugging Face en Dataset PyTorch avec transforms"""
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.classes = hf_dataset.features['label'].names
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        # Cache des labels pour éviter de recharger
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


class ConcatDataset(torch.utils.data.Dataset):
    """Combine plusieurs datasets"""
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
    """DataModule pour fake/real detection - charge depuis Hugging Face"""

    def __init__(self, hf_repo_id, batch_size=32, aug_strength=0.5, token=None):
        super().__init__()
        self.hf_repo_id = hf_repo_id
        self.batch_size = batch_size
        self.token = token

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
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15 * aug_strength, scale=(0.02, 0.2))
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
        dataset_repo = "julienlucas/midjourney-dalle-sd-nanobananapro-dataset"

        print(f"Chargement {dataset_repo} depuis Hugging Face...")
        ds = load_dataset(dataset_repo, token=self.token)

        self.train_dataset = HFDatasetWrapper(ds['train'], self.train_transform)
        self.val_dataset = HFDatasetWrapper(ds['test'], self.val_transform)

        # Calculer les poids depuis le dataset HF directement
        all_labels = [item['label'] for item in ds['train']]

        class_counts = Counter(all_labels)
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        # Correction: utiliser l'index de classe comme clé, pas enumerate
        self.class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]

        print(f"Class weights: {self.class_weights}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        # Utiliser le cache des labels depuis le wrapper
        if hasattr(self.train_dataset, '_labels'):
            all_labels = self.train_dataset._labels
        else:
            all_labels = [item['label'] for item in self.train_dataset.hf_dataset]

        weights = [self.class_weights[label] for label in all_labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )


class AttentionBlock(nn.Module):
    """Bloc d'attention pour améliorer la précision"""
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x)
        return x * attn


class OptunaFakeDetector(pl.LightningModule):
    """Classifier ResNet18 optimisé pour fake/real detection avec architecture améliorée"""

    def __init__(self, trial, num_classes=2, class_weights=None, pretrained_weights_path=None):
        super().__init__()

        # ============================================================
        # HYPERPARAMÈTRES À OPTIMISER (8-10 critiques)
        # ============================================================
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        lr_ratio = trial.suggest_float("lr_ratio", 0.01, 0.15, log=True)

        dropout1 = trial.suggest_float("dropout1", 0.15, 0.45)
        dropout2 = trial.suggest_float("dropout2", 0.1, 0.35)

        scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "onecycle", "cosine_warmup"])
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 2.5)
        mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.35)
        batch_size_factor = trial.suggest_categorical("batch_size_factor", [1, 2])  # Pour ajuster accumulate_grad

        # ============================================================
        # HYPERPARAMÈTRES FIXÉS (valeurs safe)
        # ============================================================
        hidden_size1 = 512
        hidden_size2 = 256
        use_attention = False
        use_batchnorm = True
        dropout3 = 0.0
        label_smoothing = 0.05
        unfreeze_epoch = 3
        num_unfreeze_blocks = 2  # layer4 + layer3

        self.save_hyperparameters()
        self.trial = trial
        self.unfreeze_epoch = unfreeze_epoch
        self.num_unfreeze_blocks = num_unfreeze_blocks
        self.lr_ratio = lr_ratio
        self.label_smoothing = label_smoothing
        self.scheduler_type = scheduler_type
        self.focal_gamma = focal_gamma
        self.use_attention = use_attention
        self.mixup_alpha = mixup_alpha
        self.batch_size_factor = batch_size_factor

        print(f"Trial {trial.number}: Chargement modèle pré-entraîné {pretrained_weights_path}...")
        model = tv_models.resnet18(weights=None)
        num_ftrs = model.fc.in_features

        state_dict = torch.load(pretrained_weights_path, map_location='cpu', weights_only=True)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        model_state = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                        if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered_dict, strict=False)
        print(f"Trial {trial.number}: Backbone pré-entraîné chargé")

        # Architecture fixée: 512 -> 256 -> 2 avec BatchNorm
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden_size2, num_classes)
        )

        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        self.model = model

        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        if focal_gamma > 0:
            self.loss_fn = FocalLoss(alpha=class_weights_tensor, gamma=focal_gamma, label_smoothing=label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

        # Métriques par classe (fake=0, real=1 typiquement)
        self.precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.mixup_alpha > 0 and self.training:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=x.device)
            x = lam * x + (1 - lam) * x[index]
            logits = self(x)
            loss = lam * self.loss_fn(logits, y) + (1 - lam) * self.loss_fn(logits, y[index])
        else:
            logits = self(x)
            loss = self.loss_fn(logits, y)

        if self.current_epoch == self.unfreeze_epoch and batch_idx == 0:
            layers_to_unfreeze = ["fc"]
            if self.num_unfreeze_blocks >= 1:
                layers_to_unfreeze.append("layer4")
            if self.num_unfreeze_blocks >= 2:
                layers_to_unfreeze.append("layer3")
            if self.num_unfreeze_blocks >= 3:
                layers_to_unfreeze.append("layer2")

            unfrozen_count = 0
            for name, param in self.model.named_parameters():
                for layer_name in layers_to_unfreeze:
                    if layer_name in name:
                        param.requires_grad = True
                        unfrozen_count += 1
            print(f"Epoch {self.current_epoch}: Dégelé {unfrozen_count} params des layers {layers_to_unfreeze}")

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1(logits, y)

        # Métriques par classe
        precision_per_class = self.precision_per_class(logits, y)
        recall_per_class = self.recall_per_class(logits, y)

        metrics = {
            'val_loss': loss,
            'val_acc': acc,
            'val_f1': f1,
        }

        # Ajouter métriques par classe (fake=0, real=1)
        class_names = ['fake', 'real']
        for i, name in enumerate(class_names[:self.num_classes]):
            metrics[f'val_precision_{name}'] = precision_per_class[i]
            metrics[f'val_recall_{name}'] = recall_per_class[i]

        self.log_dict(metrics, on_step=False, on_epoch=True)

        return {'val_acc': acc, 'val_f1': f1}

    def configure_optimizers(self):
        backbone_params = [p for n, p in self.model.named_parameters() if 'fc' not in n]
        head_params = [p for n, p in self.model.named_parameters() if 'fc' in n]

        optimizer = optim.AdamW(
            [
                {'params': backbone_params, 'lr': self.learning_rate * self.lr_ratio},
                {'params': head_params, 'lr': self.learning_rate}
            ],
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        if self.scheduler_type == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[self.learning_rate * self.lr_ratio, self.learning_rate],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.2,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=100.0
            )
            interval = "step"
            monitor = None
        elif self.scheduler_type == "cosine_warmup":
            from torch.optim.lr_scheduler import LambdaLR
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(0.1 * total_steps)

            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

            scheduler = LambdaLR(optimizer, lr_lambda)
            interval = "step"
            monitor = None
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7
            )
            interval = "epoch"
            monitor = None

        config = {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": interval}}
        if monitor:
            config["lr_scheduler"]["monitor"] = monitor
        return config


class FocalLoss(nn.Module):
    """Focal Loss pour gérer les cas difficiles"""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        alpha = self.alpha.to(inputs.device) if self.alpha is not None else None
        ce_loss = F.cross_entropy(inputs, targets, weight=alpha, label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class OptunaPruningCallback(pl.Callback):
    """Callback pour pruner les essais Optuna"""

    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        current_score = trainer.callback_metrics.get('val_acc')

        if current_score is not None:
            self.trial.report(current_score.item(), epoch)

            if self.trial.should_prune():
                raise optuna.TrialPruned()


def get_tta_transforms():
    """Retourne les transformations TTA (Test Time Augmentation)"""
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tta_transforms = [
        # Original
        base_transform,
        # Flip horizontal
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Rotation légère +5
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Rotation légère -5
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=(-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Scale légèrement différent
        transforms.Compose([
            transforms.Resize((270, 270)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ]
    return tta_transforms


def evaluate_with_tta(model, val_loader, device, num_tta=5):
    """Évalue le modèle avec TTA (Test Time Augmentation)"""
    model.eval()
    tta_transforms = get_tta_transforms()[:num_tta]

    all_preds = []
    all_labels = []

    # Récupérer le dataset original
    dataset = val_loader.dataset
    if hasattr(dataset, 'hf_dataset'):
        hf_dataset = dataset.hf_dataset
    else:
        hf_dataset = dataset

    with torch.no_grad():
        for idx in range(len(hf_dataset)):
            item = hf_dataset[idx]
            image = item['image'] if isinstance(item, dict) else item[0]
            label = item['label'] if isinstance(item, dict) else item[1]

            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Appliquer chaque transformation TTA et moyenner les prédictions
            tta_logits = []
            for transform in tta_transforms:
                img_tensor = transform(image).unsqueeze(0).to(device)
                logits = model(img_tensor)
                tta_logits.append(F.softmax(logits, dim=1))

            # Moyenne des probabilités
            avg_probs = torch.stack(tta_logits).mean(dim=0)
            pred = avg_probs.argmax(dim=1).item()

            all_preds.append(pred)
            all_labels.append(label)

    # Calcul des métriques
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    accuracy = (all_preds == all_labels).float().mean().item()

    # Métriques par classe
    num_classes = 2
    metrics = {'tta_accuracy': accuracy}

    class_names = ['fake', 'real']
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).float().mean().item()
            metrics[f'tta_acc_{name}'] = class_acc

            # Precision et recall par classe
            tp = ((all_preds == i) & (all_labels == i)).sum().item()
            fp = ((all_preds == i) & (all_labels != i)).sum().item()
            fn = ((all_preds != i) & (all_labels == i)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            metrics[f'tta_precision_{name}'] = precision
            metrics[f'tta_recall_{name}'] = recall

    return metrics


def save_model_to_hf(model_state_dict, trial_number, model_name, accuracy, hyperparams, api, repo_id, token):
    """Sauvegarde le modèle sur HuggingFace"""
    import tempfile

    # Nom du fichier: trial_X_nomdumodel_precision.pth
    filename = f"trial_{trial_number}_{model_name}_{accuracy:.4f}.pth"

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        torch.save({
            'model_state_dict': model_state_dict,
            'trial_number': trial_number,
            'accuracy': accuracy,
            'hyperparams': hyperparams
        }, tmp.name)

        try:
            api.upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=f"models/{filename}",
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"✅ Modèle uploadé: {repo_id}/models/{filename}")
        except Exception as e:
            print(f"⚠️  Erreur upload modèle: {e}")
        finally:
            os.unlink(tmp.name)


def update_csv_on_hf(study, api, repo_id, token):
    """Met à jour le CSV des résultats sur HuggingFace"""
    import tempfile

    df = study.trials_dataframe()
    csv_path = "./optuna_results_fake_detector.csv"
    df.to_csv(csv_path, index=False)

    try:
        api.upload_file(
            path_or_fileobj=csv_path,
            path_in_repo="optuna_results_fake_detector.csv",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f"✅ CSV mis à jour sur {repo_id}")
    except Exception as e:
        print(f"⚠️  Erreur upload CSV: {e}")


def objective(trial, hf_repo_id, pretrained_path, token=None, study=None):
    """Fonction objectif pour Optuna"""
    global BEST_MODELS, HF_API, HF_OUTPUT_REPO, HF_TOKEN

    pl.seed_everything(42)

    try:
        # Paramètres data FIXÉS pour réduire l'espace de recherche
        batch_size = 32  # Fixé
        aug_strength = 0.6  # Fixé

        print(f"[Trial {trial.number}] batch_size={batch_size} (fixé), aug_strength={aug_strength:.2f} (fixé)")

        print(f"[Trial {trial.number}] Chargement des données depuis Hugging Face...")
        dm = FakeDetectorDataModule(hf_repo_id, batch_size=batch_size, aug_strength=aug_strength, token=token)
        dm.setup()
        print(f"[Trial {trial.number}] Données chargées ({len(dm.train_dataset)} train, {len(dm.val_dataset)} val)")

        print(f"[Trial {trial.number}] Création du modèle...")
        model = OptunaFakeDetector(
            trial=trial,
            num_classes=2,
            class_weights=dm.class_weights,
            pretrained_weights_path=pretrained_path
        )
        print(f"[Trial {trial.number}] Modèle créé")

        early_stop = EarlyStopping(
            monitor="val_acc",
            patience=8,
            mode="max",
            min_delta=0.005
        )

        optuna_callback = OptunaPruningCallback(trial)

        # CUDA uniquement, pas de MPS
        if torch.cuda.is_available():
            accelerator = "gpu"
            precision = "16-mixed"
            device = torch.device("cuda")
            print(f"[Trial {trial.number}] Utilisation de CUDA")
        else:
            accelerator = "cpu"
            precision = "32"
            device = torch.device("cpu")
            print(f"[Trial {trial.number}] Utilisation de CPU (CUDA non disponible)")

        print(f"[Trial {trial.number}] Configuration du trainer...")

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        print(f"[Trial {trial.number}] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        test_batch = next(iter(train_loader))
        print(f"[Trial {trial.number}] Batch test OK: {test_batch[0].shape}, {test_batch[1].shape}")

        # Récupérer batch_size_factor du modèle pour accumulate_grad_batches
        accumulate_grad = model.batch_size_factor if hasattr(model, 'batch_size_factor') else 2

        progress_cb = TQDMProgressBar(refresh_rate=1)
        trainer = pl.Trainer(
            max_epochs=20,  # Réduit de 30 à 20 pour tenir en <10h
            accelerator=accelerator,
            devices=1,
            precision=precision,
            callbacks=[early_stop, optuna_callback, progress_cb],
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            accumulate_grad_batches=accumulate_grad,
            num_sanity_val_steps=0,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            log_every_n_steps=20
        )

        print(f"[Trial {trial.number}] Démarrage de l'entraînement...")
        trainer.fit(model, dm)
        print(f"[Trial {trial.number}] Entraînement terminé")

        val_acc = trainer.callback_metrics.get('val_acc')
        val_f1 = trainer.callback_metrics.get('val_f1')

        # Métriques par classe
        precision_fake = trainer.callback_metrics.get('val_precision_fake')
        recall_fake = trainer.callback_metrics.get('val_recall_fake')
        precision_real = trainer.callback_metrics.get('val_precision_real')
        recall_real = trainer.callback_metrics.get('val_recall_real')

        if precision_fake is not None:
            print(f"[Trial {trial.number}] Métriques par classe:")
            print(f"  FAKE  - Precision: {precision_fake.item():.4f}, Recall: {recall_fake.item():.4f}")
            print(f"  REAL  - Precision: {precision_real.item():.4f}, Recall: {recall_real.item():.4f}")

        if val_acc is not None and val_f1 is not None:
            acc = val_acc.item()
            f1 = val_f1.item()
            score = 0.7 * acc + 0.3 * f1
            print(f"[Trial {trial.number}] Acc: {acc:.4f}, F1: {f1:.4f}, Score combiné: {score:.4f}")

            # Sauvegarder le modèle sur HuggingFace
            if HF_API is not None and HF_OUTPUT_REPO is not None:
                model_state = copy.deepcopy(model.model.state_dict())
                hyperparams = trial.params.copy()

                # Upload le modèle
                save_model_to_hf(
                    model_state,
                    trial.number,
                    "resnet18_fakedetector",
                    acc,
                    hyperparams,
                    HF_API,
                    HF_OUTPUT_REPO,
                    HF_TOKEN
                )

                # Garder les meilleurs modèles en mémoire
                BEST_MODELS.append((score, trial.number, model_state, hyperparams))
                BEST_MODELS.sort(reverse=True, key=lambda x: x[0])
                BEST_MODELS = BEST_MODELS[:MAX_BEST_MODELS]

                # Mettre à jour le CSV sur HuggingFace
                if study is not None:
                    update_csv_on_hf(study, HF_API, HF_OUTPUT_REPO, HF_TOKEN)

            return score
        elif val_acc is not None:
            score = val_acc.item()
            print(f"[Trial {trial.number}] Score final (acc only): {score:.4f}")
            return score
        print(f"[Trial {trial.number}] Aucun score trouvé, retour 0.0")
        return 0.0
    except optuna.TrialPruned:
        print(f"[Trial {trial.number}] Trial pruné (arrêté car non prometteur)")
        raise
    except Exception as e:
        print(f"[Trial {trial.number}] ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def create_ensemble_model(best_models, num_classes=2):
    """Crée un modèle ensemble à partir des meilleurs modèles"""
    class EnsembleModel(nn.Module):
        def __init__(self, model_states, num_classes):
            super().__init__()
            self.models = nn.ModuleList()
            for state_dict in model_states:
                model = tv_models.resnet18(weights=None)
                # Recréer l'architecture de la tête (simplifié - Linear standard)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
                # Charger les poids du backbone uniquement (la tête peut différer)
                model_state = model.state_dict()
                filtered = {k: v for k, v in state_dict.items()
                           if k in model_state and model_state[k].shape == v.shape}
                model.load_state_dict(filtered, strict=False)
                self.models.append(model)

        def forward(self, x):
            outputs = [F.softmax(model(x), dim=1) for model in self.models]
            return torch.stack(outputs).mean(dim=0)

    model_states = [m[2] for m in best_models]  # (score, trial_num, state_dict, hyperparams)
    return EnsembleModel(model_states, num_classes)


def run_optuna_optimization(n_trials=50, n_jobs=1, hf_repo_id=None, pretrained_path=None, hf_output_repo="julienlucas/fakefinder", token=None):
    """Lance l'optimisation Optuna et upload le CSV sur Hugging Face"""
    global HF_API, HF_OUTPUT_REPO, HF_TOKEN, BEST_MODELS

    # Initialiser les variables globales pour l'upload continu
    if not token:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_ACCESS_TOKEN")
    if token:
        token = token.strip() if isinstance(token, str) else token
        login(token=token)
        HF_API = HfApi(token=token)
        HF_OUTPUT_REPO = hf_output_repo
        HF_TOKEN = token
        print(f"✅ Connexion HuggingFace établie - Upload continu activé vers {hf_output_repo}")
    else:
        print("⚠️  Token HF non trouvé - Upload continu désactivé")
        HF_API = None

    BEST_MODELS = []  # Reset la liste des meilleurs modèles

    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=7,  # Réduit de 10 à 7 (on a 35 trials, garde 28 pour optimisation)
        multivariate=True,
    )

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=4,   # Réduit de 5 à 4 pour 20 epochs
        max_resource=20,  # Aligné avec max_epochs=20
        reduction_factor=3
    )

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name="fake_detector"
    )

    print(f"\nDémarrage de l'optimisation avec {n_trials} essais...")
    print("Dataset Hugging Face: julienlucas/midjourney-dalle-sd-nanobananapro-dataset")
    print("Objectif: Passer de 85-87% à >95%")
    print("Stratégie: Fine-tuning progressif + Mixup + Focal Loss + Hyperband pruning")
    print("CUDA uniquement (MPS désactivé)")
    print("Config optimisée: 9 HPs à optimiser, max_epochs=20, ~9h estimé sur A10")
    print()

    # Passer le study pour pouvoir mettre à jour le CSV à chaque trial
    study.optimize(
        lambda trial: objective(trial, hf_repo_id, pretrained_path, token, study=study),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    print("\n" + "="*50)
    print("OPTIMISATION TERMINÉE")
    print("="*50)

    best_trial = study.best_trial
    print(f"\nMeilleure précision: {best_trial.value:.4f}")
    print("\nMeilleurs hyperparamètres:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Sauvegarde finale du CSV
    df = study.trials_dataframe()
    csv_path = "./optuna_results_fake_detector.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés dans {csv_path}")

    # Upload final du CSV
    if HF_API is not None:
        update_csv_on_hf(study, HF_API, HF_OUTPUT_REPO, HF_TOKEN)

    # Afficher les top 5 meilleurs modèles
    print("\n" + "="*50)
    print(f"TOP {len(BEST_MODELS)} MEILLEURS MODÈLES")
    print("="*50)
    for i, (score, trial_num, _, hyperparams) in enumerate(BEST_MODELS):
        print(f"\n#{i+1} - Trial {trial_num}: Score = {score:.4f}")
        print(f"    LR: {hyperparams.get('learning_rate', 'N/A'):.6f}")
        print(f"    Scheduler: {hyperparams.get('scheduler', 'N/A')}")
        print(f"    Dropout1: {hyperparams.get('dropout1', 'N/A'):.3f}")

    return study, best_trial


def run_final_evaluation_with_tta(hf_repo_id, token=None):
    """Évalue les meilleurs modèles avec TTA et crée un ensemble"""
    global BEST_MODELS

    if len(BEST_MODELS) == 0:
        print("⚠️  Aucun modèle sauvegardé pour l'évaluation finale")
        return

    print("\n" + "="*60)
    print("ÉVALUATION FINALE AVEC TTA (Test Time Augmentation)")
    print("="*60)

    # Charger les données de validation
    print("\nChargement des données de validation...")
    dm = FakeDetectorDataModule(hf_repo_id, batch_size=32, aug_strength=0.5, token=token)
    dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Évaluer chaque modèle avec TTA
    print("\n--- Évaluation individuelle avec TTA ---")
    for i, (score, trial_num, state_dict, hyperparams) in enumerate(BEST_MODELS):
        print(f"\nModèle #{i+1} (Trial {trial_num}, Score original: {score:.4f})")

        # Reconstruire le modèle
        model = tv_models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

        # Charger les poids compatibles
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                   if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        model = model.to(device)
        model.eval()

        # TTA evaluation
        tta_metrics = evaluate_with_tta(model, dm.val_dataloader(), device, num_tta=5)
        print(f"  TTA Accuracy: {tta_metrics['tta_accuracy']:.4f}")
        print(f"  FAKE  - Precision: {tta_metrics.get('tta_precision_fake', 0):.4f}, Recall: {tta_metrics.get('tta_recall_fake', 0):.4f}")
        print(f"  REAL  - Precision: {tta_metrics.get('tta_precision_real', 0):.4f}, Recall: {tta_metrics.get('tta_recall_real', 0):.4f}")

    # Évaluer l'ensemble
    if len(BEST_MODELS) >= 2:
        print("\n--- Évaluation de l'Ensemble (Top 5 modèles) ---")
        ensemble = create_ensemble_model(BEST_MODELS, num_classes=2)
        ensemble = ensemble.to(device)
        ensemble.eval()

        tta_metrics = evaluate_with_tta(ensemble, dm.val_dataloader(), device, num_tta=5)
        print(f"  Ensemble TTA Accuracy: {tta_metrics['tta_accuracy']:.4f}")
        print(f"  FAKE  - Precision: {tta_metrics.get('tta_precision_fake', 0):.4f}, Recall: {tta_metrics.get('tta_recall_fake', 0):.4f}")
        print(f"  REAL  - Precision: {tta_metrics.get('tta_precision_real', 0):.4f}, Recall: {tta_metrics.get('tta_recall_real', 0):.4f}")

        # Sauvegarder l'ensemble sur HuggingFace
        if HF_API is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
                torch.save({
                    'ensemble_state_dicts': [m[2] for m in BEST_MODELS],
                    'ensemble_trial_numbers': [m[1] for m in BEST_MODELS],
                    'ensemble_scores': [m[0] for m in BEST_MODELS],
                    'tta_accuracy': tta_metrics['tta_accuracy']
                }, tmp.name)

                try:
                    HF_API.upload_file(
                        path_or_fileobj=tmp.name,
                        path_in_repo=f"models/ensemble_top{len(BEST_MODELS)}_{tta_metrics['tta_accuracy']:.4f}.pth",
                        repo_id=HF_OUTPUT_REPO,
                        repo_type="model",
                        token=HF_TOKEN
                    )
                    print(f"✅ Ensemble uploadé sur HuggingFace")
                except Exception as e:
                    print(f"⚠️  Erreur upload ensemble: {e}")
                finally:
                    os.unlink(tmp.name)


if __name__ == "__main__":
    hf_output_repo = "julienlucas/fakefinder"
    token = os.getenv("HF_ACCESS_TOKEN")

    print(f"📥 Téléchargement du modèle depuis {hf_output_repo}...")
    pretrained_path = hf_hub_download(
        repo_id="julienlucas/fakefinder",
        filename="best_76_resnet18_fake_detector.pth",
        repo_type="model",
        token=token
    )
    print(f"✅ Modèle téléchargé: {pretrained_path}")

    study, best_trial = run_optuna_optimization(
        n_trials=35,  # Réduit de 50 à 35 pour tenir en <10h
        n_jobs=1,
        hf_repo_id="julienlucas/fakefinder",
        pretrained_path=pretrained_path,
        hf_output_repo="julienlucas/fakefinder",
        token=token
    )

    print("\n" + "="*60)
    print("MEILLEURS HYPERPARAMÈTRES TROUVÉS")
    print("="*60)
    print(f"\nPrécision atteinte: {best_trial.value:.4f} ({best_trial.value*100:.2f}%)")
    print("\nHyperparamètres optimaux:")
    print("-"*40)

    categories = {
        "Optimisation": ["learning_rate", "weight_decay", "lr_ratio", "scheduler"],
        "Régularisation": ["dropout1", "dropout2", "focal_gamma", "mixup_alpha"],
        "Training": ["batch_size_factor"]
    }

    print("\n[Paramètres FIXÉS]")
    print("  hidden_size1 = 512, hidden_size2 = 256")
    print("  use_batchnorm = True, use_attention = False")
    print("  label_smoothing = 0.05, dropout3 = 0.0")
    print("  unfreeze_epoch = 3, num_unfreeze_blocks = 2")
    print("  batch_size = 32, aug_strength = 0.6")

    for cat_name, params in categories.items():
        print(f"\n{cat_name}:")
        for key in params:
            if key in best_trial.params:
                value = best_trial.params[key]
                if isinstance(value, float):
                    print(f"  {key} = {value:.6f}")
                else:
                    print(f"  {key} = {value}")

    # Évaluation finale avec TTA et Ensemble
    run_final_evaluation_with_tta(
        hf_repo_id="julienlucas/midjourney-dalle-sd-nanobananapro-dataset",
        token=token
    )