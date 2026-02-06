import os
from dotenv import load_dotenv

# Charger le .env depuis le répertoire racine du projet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, ".env"))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import io
import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar, StochasticWeightAveraging
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
from huggingface_hub import HfApi, hf_hub_download, login
from PIL import Image, ImageEnhance
import copy
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader

# Global pour stocker les meilleurs modèles (top 5)
BEST_MODELS = []  # Liste de (score, trial_number, model_state_dict, hyperparams)
MAX_BEST_MODELS = 5

# Global pour l'API HuggingFace et le repo
HF_API = None
HF_OUTPUT_REPO = None
HF_TOKEN = None


# ============================================================
# AUGMENTATIONS DE TEXTURE POUR DÉTECTER LES ARTEFACTS GAN/DIFFUSION
# ============================================================

class RandomJPEGCompression:
    """
    Simule la compression JPEG des réseaux sociaux.
    Crucial pour révéler les artefacts des images générées.
    """
    def __init__(self, probability=0.3, quality_range=(40, 90)):
        self.probability = probability
        self.quality_range = quality_range

    def __call__(self, img):
        if torch.rand(1).item() < self.probability:
            output = io.BytesIO()
            quality = torch.randint(self.quality_range[0], self.quality_range[1], (1,)).item()
            img.save(output, format='JPEG', quality=quality)
            output.seek(0)
            return Image.open(output).convert('RGB')
        return img


class RandomSharpness:
    """
    Variation de netteté pour forcer le modèle à détecter les textures.
    Les IA génératives laissent des traces dans les hautes fréquences.
    """
    def __init__(self, factor_range=(0.5, 2.5), probability=0.2):
        self.factor_range = factor_range
        self.probability = probability

    def __call__(self, img):
        if torch.rand(1).item() < self.probability:
            factor = np.random.uniform(*self.factor_range)
            return ImageEnhance.Sharpness(img).enhance(factor)
        return img


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

    def __init__(self, hf_repo_id, batch_size=16, aug_strength=0.5, token=None):
        super().__init__()
        self.hf_repo_id = hf_repo_id
        self.batch_size = batch_size
        self.token = token

        # EfficientNet-V2-S : résolution 384x384 (non-négociable)
        # Augmentations de texture cruciales pour GAN/Diffusion
        self.train_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            # Augmentations de texture AVANT le crop (crucial!)
            RandomJPEGCompression(probability=0.3, quality_range=(40, 90)),
            RandomSharpness(factor_range=(0.5, 2.5), probability=0.2),
            transforms.RandomResizedCrop(384, scale=(0.75, 1.0)),
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
            transforms.Resize((384, 384)),
            transforms.CenterCrop(384),
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

        all_labels = [item['label'] for item in ds['train']]

        class_counts = Counter(all_labels)
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        self.class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]

        print(f"Class weights: {self.class_weights}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def train_dataloader(self):
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
    """Focal Loss pour gérer les cas difficiles - gamma=2.0 standard industrie"""
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


class OptunaEfficientNetDetector(pl.LightningModule):
    """
    EfficientNet-V2-S optimisé pour fake/real detection avec:
    - Pooling Mixte (Avg + Max) pour capturer les artefacts
    - Correction du bug de dégel
    - SWA compatible
    """

    def __init__(self, trial, num_classes=2, class_weights=None):
        super().__init__()

        # ============================================================
        # HYPERPARAMÈTRES À OPTIMISER (plages ajustées pour 95%)
        # ============================================================
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        lr_ratio = trial.suggest_float("lr_ratio", 0.01, 0.2, log=True)

        dropout = trial.suggest_float("dropout", 0.3, 0.5)

        scheduler_type = trial.suggest_categorical("scheduler", ["onecycle", "cosine_warmup"])

        # Focal gamma fixé à 2.0 (standard industrie)
        focal_gamma = 2.0

        # Label smoothing max 0.05 (trop masque les erreurs subtiles)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.05)

        # Nombre de blocs à dégeler (1-4 pour garder généralisation)
        unfreeze_blocks = trial.suggest_int("unfreeze_blocks", 1, 4)
        unfreeze_epoch = trial.suggest_int("unfreeze_epoch", 2, 4)

        # Accumulation de gradients (4 ou 8 pour stabiliser BatchNorm)
        accumulate_grad = trial.suggest_categorical("accumulate_grad", [4, 8])

        self.save_hyperparameters()
        self.trial = trial
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_blocks = unfreeze_blocks
        self.lr_ratio = lr_ratio
        self.label_smoothing = label_smoothing
        self.scheduler_type = scheduler_type
        self.focal_gamma = focal_gamma
        self.accumulate_grad = accumulate_grad

        # ============================================================
        # ARCHITECTURE: EfficientNet-V2-S + Pooling Mixte
        # ============================================================
        print(f"Trial {trial.number}: Chargement EfficientNet-V2-S (ImageNet weights)...")
        model = tv_models.efficientnet_v2_s(weights=tv_models.EfficientNet_V2_S_Weights.DEFAULT)

        # Récupérer le nombre de features avant de modifier
        num_ftrs = model.classifier[1].in_features
        self.num_ftrs = num_ftrs

        # Enlever le classifier par défaut
        model.classifier = nn.Identity()

        # Pooling Mixte: capture les indices moyens ET les pics extrêmes
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

        # Nouveau classifier avec features concaténées (Avg + Max)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs * 2, 512),  # *2 car on concatène Avg et Max
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

        # Geler tout le backbone au début
        for param in model.features.parameters():
            param.requires_grad = False

        # Le classifier custom est entraînable dès le départ
        for param in self.classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.classifier.parameters())
        total = sum(p.numel() for p in model.parameters()) + trainable
        print(f"Trial {trial.number}: Parametres entrainables: {trainable:,} / {total:,} (classifier)")

        self.model = model

        # Loss function
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.loss_fn = FocalLoss(alpha=class_weights_tensor, gamma=focal_gamma, label_smoothing=label_smoothing)

        # Métriques
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes

    def forward(self, x):
        # Extraction des features via le backbone
        features = self.model.features(x)

        # Pooling mixte : moyenne + max
        avg_pool = self.pool_avg(features)
        max_pool = self.pool_max(features)

        # Concaténation des deux représentations
        combined = torch.cat([avg_pool, max_pool], dim=1)

        # Classification
        return self.classifier(combined)

    def _unfreeze_blocks(self):
        """Dégèle les N derniers blocs de features"""
        if self.unfreeze_blocks == 0:
            return 0

        # EfficientNet-V2-S a 8 éléments dans features (0=stem, 1-7=blocs)
        total_blocks = len(self.model.features)
        blocks_to_unfreeze = list(range(total_blocks - self.unfreeze_blocks, total_blocks))

        unfrozen_count = 0
        for idx in blocks_to_unfreeze:
            for param in self.model.features[idx].parameters():
                param.requires_grad = True
                unfrozen_count += 1

        return unfrozen_count

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Dégeler les blocs à l'époque prévue
        if self.current_epoch == self.unfreeze_epoch and batch_idx == 0:
            unfrozen = self._unfreeze_blocks()
            if unfrozen > 0:
                print(f"Epoch {self.current_epoch}: Degele {unfrozen} params ({self.unfreeze_blocks} derniers blocs)")

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = self.accuracy(logits, y)
        f1 = self.f1(logits, y)
        precision_per_class = self.precision_per_class(logits, y)
        recall_per_class = self.recall_per_class(logits, y)

        metrics = {
            'val_loss': loss,
            'val_acc': acc,
            'val_f1': f1,
        }

        class_names = ['fake', 'real']
        for i, name in enumerate(class_names[:self.num_classes]):
            metrics[f'val_precision_{name}'] = precision_per_class[i]
            metrics[f'val_recall_{name}'] = recall_per_class[i]

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_acc': acc, 'val_f1': f1}

    def configure_optimizers(self):
        # ============================================================
        # CORRECTION DU BUG DE DÉGEL: On inclut TOUS les paramètres
        # même ceux avec requires_grad=False au début
        # ============================================================
        backbone_params = []
        head_params = []

        # Paramètres du backbone (features)
        for param in self.model.features.parameters():
            backbone_params.append(param)

        # Paramètres du classifier custom
        for param in self.classifier.parameters():
            head_params.append(param)

        # On donne TOUS les paramètres à l'optimiseur
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.learning_rate * self.lr_ratio},
            {'params': head_params, 'lr': self.learning_rate}
        ], weight_decay=self.weight_decay)

        # OneCycleLR avec warmup (crucial pour EfficientNet)
        if self.scheduler_type == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[self.learning_rate * self.lr_ratio, self.learning_rate],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=100.0
            )
            interval = "step"
        else:  # cosine_warmup
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(0.1 * total_steps)

            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            interval = "step"

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": interval}}


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


def save_model_to_hf(model, trial_number, accuracy, hyperparams, api, repo_id, token):
    """Sauvegarde le modèle complet (backbone + classifier) sur HuggingFace"""
    import tempfile

    filename = f"trial_{trial_number}_efficientnetv2s_{accuracy:.4f}.pth"

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'classifier_state_dict': model.classifier.state_dict(),
            'pool_avg': model.pool_avg.state_dict(),
            'pool_max': model.pool_max.state_dict(),
            'num_ftrs': model.num_ftrs,
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
            print(f"  Modele uploade: {repo_id}/models/{filename}")
        except Exception as e:
            print(f"  Erreur upload modele: {e}")
        finally:
            os.unlink(tmp.name)


def update_csv_on_hf(study, api, repo_id, token):
    """Met à jour le CSV des résultats sur HuggingFace"""
    df = study.trials_dataframe()
    csv_path = "./optuna_efficientnet_results.csv"
    df.to_csv(csv_path, index=False)

    try:
        api.upload_file(
            path_or_fileobj=csv_path,
            path_in_repo="optuna_efficientnet_results.csv",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f"  CSV mis a jour sur {repo_id}")
    except Exception as e:
        print(f"  Erreur upload CSV: {e}")


def objective(trial, hf_repo_id, token=None, study=None):
    """Fonction objectif pour Optuna - EfficientNet-V2-S avec toutes les optimisations"""
    global BEST_MODELS, HF_API, HF_OUTPUT_REPO, HF_TOKEN

    pl.seed_everything(42)

    try:
        batch_size = 16  # Fixé pour éviter OOM sur A10 avec 384x384
        aug_strength = 0.6

        print(f"\n{'='*60}")
        print(f"[Trial {trial.number}] Demarrage")
        print(f"{'='*60}")

        dm = FakeDetectorDataModule(hf_repo_id, batch_size=batch_size, aug_strength=aug_strength, token=token)
        dm.setup()

        model = OptunaEfficientNetDetector(
            trial=trial,
            num_classes=2,
            class_weights=dm.class_weights
        )

        # Early stopping avec patience=4 (laisser le dégel agir)
        early_stop = EarlyStopping(
            monitor="val_acc",
            patience=4,
            mode="max",
            min_delta=0.003
        )

        optuna_callback = OptunaPruningCallback(trial)

        # SWA pour stabiliser les derniers pourcents (commence à 70% de l'entraînement)
        swa_callback = StochasticWeightAveraging(
            swa_lrs=1e-5,
            swa_epoch_start=0.7
        )

        if torch.cuda.is_available():
            accelerator = "gpu"
            # bf16-mixed si GPU Ampere+ (A10 supporte)
            if torch.cuda.get_device_capability()[0] >= 8:
                precision = "bf16-mixed"
                print(f"[Trial {trial.number}] GPU CUDA Ampere+ - bf16-mixed")
            else:
                precision = "16-mixed"
                print(f"[Trial {trial.number}] GPU CUDA - 16-mixed")
        else:
            accelerator = "cpu"
            precision = "32"
            print(f"[Trial {trial.number}] CPU uniquement")

        progress_cb = TQDMProgressBar(refresh_rate=1)

        trainer = pl.Trainer(
            max_epochs=15,  # 15 epochs pour laisser le temps au SWA
            accelerator=accelerator,
            devices=1,
            precision=precision,
            callbacks=[early_stop, optuna_callback, progress_cb, swa_callback],
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            accumulate_grad_batches=model.accumulate_grad,
            num_sanity_val_steps=0,
            log_every_n_steps=20
        )

        print(f"[Trial {trial.number}] Hyperparams: lr={trial.params['learning_rate']:.6f}, "
              f"lr_ratio={trial.params['lr_ratio']:.3f}, dropout={trial.params['dropout']:.2f}, "
              f"unfreeze_blocks={trial.params['unfreeze_blocks']}, unfreeze_epoch={trial.params['unfreeze_epoch']}")

        trainer.fit(model, dm)

        val_acc = trainer.callback_metrics.get('val_acc')
        val_f1 = trainer.callback_metrics.get('val_f1')

        precision_fake = trainer.callback_metrics.get('val_precision_fake')
        recall_fake = trainer.callback_metrics.get('val_recall_fake')
        precision_real = trainer.callback_metrics.get('val_precision_real')
        recall_real = trainer.callback_metrics.get('val_recall_real')

        if precision_fake is not None:
            print(f"[Trial {trial.number}] Metriques par classe:")
            print(f"  FAKE  - Precision: {precision_fake.item():.4f}, Recall: {recall_fake.item():.4f}")
            print(f"  REAL  - Precision: {precision_real.item():.4f}, Recall: {recall_real.item():.4f}")

        if val_acc is not None and val_f1 is not None:
            acc = val_acc.item()
            f1 = val_f1.item()
            score = 0.7 * acc + 0.3 * f1
            print(f"[Trial {trial.number}] Acc: {acc:.4f}, F1: {f1:.4f}, Score: {score:.4f}")

            if HF_API is not None and HF_OUTPUT_REPO is not None:
                hyperparams = trial.params.copy()

                save_model_to_hf(
                    model,
                    trial.number,
                    acc,
                    hyperparams,
                    HF_API,
                    HF_OUTPUT_REPO,
                    HF_TOKEN
                )

                model_state = {
                    'model': copy.deepcopy(model.model.state_dict()),
                    'classifier': copy.deepcopy(model.classifier.state_dict()),
                    'num_ftrs': model.num_ftrs
                }
                BEST_MODELS.append((score, trial.number, model_state, hyperparams))
                BEST_MODELS.sort(reverse=True, key=lambda x: x[0])
                BEST_MODELS = BEST_MODELS[:MAX_BEST_MODELS]

                if study is not None:
                    update_csv_on_hf(study, HF_API, HF_OUTPUT_REPO, HF_TOKEN)

            return score
        elif val_acc is not None:
            return val_acc.item()

        return 0.0

    except optuna.TrialPruned:
        print(f"[Trial {trial.number}] Prune (non prometteur)")
        raise
    except Exception as e:
        print(f"[Trial {trial.number}] ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def run_optuna_optimization(n_trials=20, hf_repo_id=None, hf_output_repo="julienlucas/fakefinder", token=None):
    """Lance l'optimisation Optuna pour EfficientNet-V2-S - 20 trials de 15 epochs"""
    global HF_API, HF_OUTPUT_REPO, HF_TOKEN, BEST_MODELS

    if not token:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_ACCESS_TOKEN")
    if token:
        token = token.strip() if isinstance(token, str) else token
        login(token=token)
        HF_API = HfApi(token=token)
        HF_OUTPUT_REPO = hf_output_repo
        HF_TOKEN = token
        print(f"Connexion HuggingFace etablie - Upload vers {hf_output_repo}")
    else:
        print("Token HF non trouve - Upload desactive")
        HF_API = None

    BEST_MODELS = []

    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=5,
        multivariate=True,
    )

    # Pruning adapté pour 15 epochs
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=4,
        max_resource=15,
        reduction_factor=3
    )

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name="efficientnet_v2_s_95pct"
    )

    print("\n" + "="*60)
    print("OPTUNA - EfficientNet-V2-S pour 95%+")
    print("="*60)
    print(f"Trials: {n_trials}")
    print(f"Epochs/trial: 15 (patience=4, SWA active)")
    print(f"Resolution: 384x384")
    print(f"Batch: 16 + accumulate_grad=[4,8]")
    print(f"Augmentations: JPEG Compression, Sharpness, ColorJitter, GaussianBlur")
    print(f"Architecture: Pooling Mixte (Avg + Max)")
    print(f"Label smoothing: 0.0-0.05 (bas)")
    print(f"Focal gamma: 2.0 (fixe)")
    print("="*60 + "\n")

    study.optimize(
        lambda trial: objective(trial, hf_repo_id, token, study=study),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True
    )

    print("\n" + "="*60)
    print("OPTIMISATION TERMINEE")
    print("="*60)

    best_trial = study.best_trial
    print(f"\nMeilleur score: {best_trial.value:.4f}")
    print("\nMeilleurs hyperparametres:")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    df = study.trials_dataframe()
    csv_path = "./optuna_efficientnet_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResultats sauvegardes: {csv_path}")

    if HF_API is not None:
        update_csv_on_hf(study, HF_API, HF_OUTPUT_REPO, HF_TOKEN)

    print("\n" + "="*60)
    print(f"TOP {len(BEST_MODELS)} MEILLEURS MODELES")
    print("="*60)
    for i, (score, trial_num, _, hyperparams) in enumerate(BEST_MODELS):
        print(f"\n#{i+1} - Trial {trial_num}: Score = {score:.4f}")
        print(f"    lr={hyperparams.get('learning_rate', 0):.6f}, lr_ratio={hyperparams.get('lr_ratio', 0):.3f}")
        print(f"    dropout={hyperparams.get('dropout', 0):.2f}, unfreeze_blocks={hyperparams.get('unfreeze_blocks', 0)}")

    return study, best_trial


# ============================================================
# PRUNING + QUANTIFICATION INT8
# ============================================================

QUANTIZE_CONFIG = {
    "hf_repo": "julienlucas/fakefinder",
    "hf_pth_filename": "models/best_f166.6_acc92.5_efficientnetv2s_fake_detector.pth",
    "hf_dataset_repo": "julienlucas/midjourney-dalle-sd-nanobananapro-dataset",
    "image_size": 384,
    "pruning_amount": 0.2,
    "num_calibration_samples": 500,
}


class EfficientNetV2SForExport(nn.Module):
    """Architecture identique à train_efficient_v2_s.py (avec BatchNorm dans le head)."""
    def __init__(self, num_classes=2, dropout=0.34):
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


def load_pth_model(pth_path, num_classes=2):
    """Charge un modèle .pth (format train_efficient_v2_s.py)."""
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    dropout = checkpoint.get('config', {}).get('dropout', 0.34) if isinstance(checkpoint, dict) else 0.34

    model = EfficientNetV2SForExport(num_classes=num_classes, dropout=dropout)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint and 'classifier_state_dict' in checkpoint:
            model.backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.head.load_state_dict(checkpoint['classifier_state_dict'], strict=False)
            if 'pool_avg' in checkpoint:
                model.pool_avg.load_state_dict(checkpoint['pool_avg'])
            if 'pool_max' in checkpoint:
                model.pool_max.load_state_dict(checkpoint['pool_max'])
        else:
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if isinstance(state_dict, dict):
                backbone_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
                head_dict = {k.replace('head.', ''): v for k, v in state_dict.items() if k.startswith('head.')}
                if backbone_dict:
                    model.backbone.load_state_dict(backbone_dict, strict=False)
                if head_dict:
                    model.head.load_state_dict(head_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model


def prune_backbone_conv2d(model, amount=0.2):
    """Prune uniquement les Conv2d du backbone (pas le head)."""
    import torch.nn.utils.prune as prune

    parameters_to_prune = []
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    print(f"  Pruning {len(parameters_to_prune)} couches Conv2d à {amount*100:.0f}%...")
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')

    return model


def export_to_onnx(model, onnx_path, image_size=384):
    """Exporte le modèle PyTorch vers ONNX FP32."""
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Déterminer la taille des features pour le fallback pooling fixe
    with torch.no_grad():
        features = model.backbone.features(dummy_input)
        _, _, h, w = features.shape

    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    try:
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=18, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes=dynamic_axes, verbose=False
        )
        print(f"  Export ONNX FP32 (opset 18, pooling adaptatif): {onnx_path}")
        return
    except Exception as e:
        print(f"  Opset 18 échoué: {e}")

    # Fallback: pooling fixe
    print(f"  Fallback: pooling fixe kernel_size={h}")
    model.pool_avg = nn.AvgPool2d(kernel_size=h, stride=h)
    model.pool_max = nn.MaxPool2d(kernel_size=h, stride=h)

    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes=dynamic_axes, verbose=False
    )
    print(f"  Export ONNX FP32 (pooling fixe {h}x{h}): {onnx_path}")


class LazyCalibrationReader(CalibrationDataReader):
    """Charge les images à la volée (lazy) avec échantillonnage balancé (50% fake / 50% real)."""
    def __init__(self, hf_dataset_repo, num_samples=500, image_size=384, token=None):
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print(f"  Préparation de {num_samples} indices de calibration (lazy, balancé)...", flush=True)
        ds = load_dataset(hf_dataset_repo, token=token)
        self.test_ds = ds['test']

        # Échantillonnage balancé : autant de fake que de real
        labels = [item['label'] for item in self.test_ds]
        indices_by_class = {}
        for idx, label in enumerate(labels):
            indices_by_class.setdefault(label, []).append(idx)

        samples_per_class = num_samples // len(indices_by_class)
        balanced_indices = []
        for cls, cls_indices in indices_by_class.items():
            chosen = np.random.permutation(cls_indices)[:samples_per_class]
            balanced_indices.extend(chosen.tolist())
            print(f"    Classe {cls}: {len(chosen)} images", flush=True)

        np.random.shuffle(balanced_indices)
        self.indices = balanced_indices
        self.num_samples = len(balanced_indices)
        self.index = 0
        print(f"  {self.num_samples} indices prêts (chargement à la volée).", flush=True)

    def get_next(self):
        if self.index >= self.num_samples:
            return None

        idx = self.indices[self.index]
        item = self.test_ds[int(idx)]
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        tensor = self.val_transform(image).unsqueeze(0).numpy().astype(np.float32)
        self.index += 1
        if self.index % 50 == 0 or self.index == 1:
            print(f"  Calibration: {self.index}/{self.num_samples}", flush=True)
        return {"input": tensor}


def preprocess_onnx_model(input_path, output_path):
    """Pré-traite le modèle ONNX (shape inference + optimisation) avant quantification."""
    try:
        from onnxruntime.quantization import preprocess_model
        print(f"  Pré-traitement ONNX (shape inference + optimisation)...")
        preprocess_model(input_path, output_path)
        print(f"  Modèle pré-traité: {output_path}")
        return output_path
    except (ImportError, Exception) as e:
        print(f"  preprocess_model indisponible ({e}), fallback shape inference...")
        try:
            onnx_model = onnx.load(input_path)
            onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
            onnx.save(onnx_model, output_path)
            print(f"  Shape inference OK: {output_path}")
            return output_path
        except Exception as e2:
            print(f"  Shape inference échouée ({e2}), utilisation du modèle original")
            return input_path


def get_nodes_to_exclude(onnx_path):
    """Identifie les nœuds sensibles à exclure de la quantification INT8.

    Exclut:
    - Les premiers Conv (stem) : très sensibles, portent l'info basse fréquence
    - Toutes les BatchNorm : la quantification détruit leurs statistiques
    - Les 2 derniers Gemm/MatMul (Linear du head) : frontière de décision fake/real
    """
    onnx_model = onnx.load(onnx_path)
    nodes_to_exclude = []
    conv_nodes = []
    gemm_nodes = []

    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            conv_nodes.append(node.name)
        elif node.op_type == 'BatchNormalization':
            nodes_to_exclude.append(node.name)
        elif node.op_type in ('Gemm', 'MatMul'):
            gemm_nodes.append(node.name)

    # Exclure les 3 premiers Conv (stem + début du premier bloc)
    # Le stem capture les features bas niveau (bords, textures) très sensibles à la quantification
    num_stem_convs = min(3, len(conv_nodes))
    nodes_to_exclude.extend(conv_nodes[:num_stem_convs])
    print(f"    Stem: {num_stem_convs} premiers Conv exclus")

    # Les 2 derniers Gemm = les 2 Linear du head (frontière de décision)
    if len(gemm_nodes) >= 2:
        nodes_to_exclude.extend(gemm_nodes[-2:])
    elif gemm_nodes:
        nodes_to_exclude.extend(gemm_nodes)

    print(f"    Total nœuds exclus: {len(nodes_to_exclude)} (stem + BN + head)")
    return nodes_to_exclude


def evaluate_onnx_model(onnx_path, hf_dataset_repo, image_size=384, token=None):
    """Évalue l'accuracy d'un modèle ONNX sur le dataset de test."""
    print(f"  Chargement du dataset de test...", flush=True)
    ds = load_dataset(hf_dataset_repo, token=token)
    test_ds = ds['test']

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    correct = 0
    total = 0
    for i, item in enumerate(test_ds):
        image = item['image']
        label = item['label']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        tensor = val_transform(image).unsqueeze(0).numpy().astype(np.float32)
        output = session.run(None, {input_name: tensor})[0]
        pred = np.argmax(output, axis=1)[0]

        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"    Évalué: {i + 1}/{len(test_ds)} (acc: {correct/total*100:.1f}%)", flush=True)

    accuracy = correct / total * 100
    del ds, test_ds
    return accuracy


if __name__ == "__main__":
    import sys

    hf_output_repo = "julienlucas/fakefinder"
    token = os.getenv("HF_ACCESS_TOKEN")

    study, best_trial = run_optuna_optimization(
        n_trials=20,
        hf_repo_id="julienlucas/midjourney-dalle-sd-nanobananapro-dataset",
        hf_output_repo=hf_output_repo,
        token=token
    )

    print("\n" + "=" * 60)
    print("CONFIG OPTIMALE POUR ENTRAINEMENT FINAL")
    print("=" * 60)
    print("\nCopie ces valeurs dans ton script d'entrainement final:")
    print("-" * 40)
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f'"{key}": {value:.6f},')
        else:
            print(f'"{key}": {value},')
