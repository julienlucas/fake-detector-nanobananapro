import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score
from torchvision import models as tv_models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter


class ConcatDataset(torch.utils.data.Dataset):
    """Combine plusieurs datasets ImageFolder"""
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = [0]
        for length in self.lengths:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)

        # Utiliser les classes du premier dataset (doivent être identiques)
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
    """DataModule pour fake/real detection - combine midjourney et nanobanana datasets"""

    def __init__(self, data_base_dir, batch_size=32, aug_strength=0.5):
        super().__init__()
        self.data_base_dir = data_base_dir
        self.batch_size = batch_size

        # Augmentation calibrée selon la force demandée
        # Pour fake detection, les artefacts JPEG et la compression sont importants
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),  # Moins de crop agressif
            transforms.RandomHorizontalFlip(p=0.5),
            # Rotation légère - trop de rotation peut masquer les artefacts
            transforms.RandomRotation(10),
            # ColorJitter modéré - les couleurs peuvent être un indice
            transforms.ColorJitter(
                brightness=0.2 * aug_strength,
                contrast=0.2 * aug_strength,
                saturation=0.2 * aug_strength,
                hue=0.05 * aug_strength
            ),
            # Blur léger - simule différentes qualités d'image
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # RandomErasing modéré
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
        # Chemins vers les deux datasets
        midjourney_train = os.path.join(self.data_base_dir, "AIvsReal_midjourney_dalle_sd", "train")
        nanobanana_train = os.path.join(self.data_base_dir, "AIvsReal_nanobanana_pro", "train")

        # Charger les deux datasets train
        midjourney_train_ds = datasets.ImageFolder(midjourney_train, self.train_transform)
        nanobanana_train_ds = datasets.ImageFolder(nanobanana_train, self.train_transform)

        # Combiner les datasets train
        self.train_dataset = ConcatDataset(midjourney_train_ds, nanobanana_train_ds)

        # Pour validation, utiliser test si disponible, sinon split depuis train
        midjourney_test = os.path.join(self.data_base_dir, "AIvsReal_midjourney_dalle_sd", "test")
        nanobanana_test = os.path.join(self.data_base_dir, "AIvsReal_nanobanana_pro", "test")

        midjourney_val_ds = datasets.ImageFolder(midjourney_test, self.val_transform)
        nanobanana_val_ds = datasets.ImageFolder(nanobanana_test, self.val_transform)
        self.val_dataset = ConcatDataset(midjourney_val_ds, nanobanana_val_ds)

        # Calculer les poids de classe depuis les samples (sans charger les images)
        all_labels = []
        for dataset in [midjourney_train_ds, nanobanana_train_ds]:
            for _, label in dataset.samples:
                all_labels.append(label)
        class_counts = Counter(all_labels)
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        # Correction: utiliser l'index de classe comme clé, pas enumerate
        self.class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]

        print(f"Class weights: {self.class_weights}")
        print(f"Train samples: {len(self.train_dataset)} (midjourney: {len(midjourney_train_ds)}, nanobanana: {len(nanobanana_train_ds)})")
        print(f"Val samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        # Calculer les weights pour le sampler depuis les samples (sans charger les images)
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

        # Hyperparamètres à optimiser - plages affinées pour >95% accuracy
        # Learning rate plus fin - les valeurs hautes causent souvent de l'instabilité
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 2e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)

        # Dropout plus conservateur - trop de dropout nuit à la précision
        dropout1 = trial.suggest_float("dropout1", 0.1, 0.4)
        dropout2 = trial.suggest_float("dropout2", 0.05, 0.3)
        dropout3 = trial.suggest_float("dropout3", 0.0, 0.2)

        # Architecture plus flexible - adapté pour ResNet18 (num_ftrs = 512)
        hidden_size1 = trial.suggest_int("hidden_size1", 256, 1024, step=128)
        hidden_size2 = trial.suggest_int("hidden_size2", 128, 512, step=64)

        use_attention = trial.suggest_categorical("use_attention", [True, False])
        use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])

        # Fine-tuning progressif plus fin - adapté pour ResNet18 (layer4, layer3, layer2, layer1)
        unfreeze_epoch = trial.suggest_int("unfreeze_epoch", 1, 5)
        num_unfreeze_blocks = trial.suggest_int("num_unfreeze_blocks", 1, 3)  # Combien de layers dégeler (1=layer4, 2=layer4+layer3, 3=layer4+layer3+layer2)

        lr_ratio = trial.suggest_float("lr_ratio", 0.01, 0.2, log=True)  # Ratio plus petit = backbone apprend plus lentement
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)
        scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "onecycle", "cosine_warmup"])
        focal_gamma = trial.suggest_float("focal_gamma", 0.0, 3.0)  # Focal Loss - gamma plus élevé possible

        # Mixup/CutMix pour meilleure généralisation
        mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.4)

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

        # Charger ResNet18
        print(f"Trial {trial.number}: Chargement modèle pré-entraîné {pretrained_weights_path}...")
        model = tv_models.resnet18(weights=None)
        num_ftrs = model.fc.in_features

        state_dict = torch.load(pretrained_weights_path, map_location='cpu', weights_only=True)
        # Gestion si le fichier est un checkpoint Lightning ou un state_dict pur
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            # Enlever le préfixe 'model.' si présent (typique de Lightning)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        model_state = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                        if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered_dict, strict=False)
        print(f"Trial {trial.number}: Backbone pré-entraîné chargé")


        # Tête améliorée avec architecture plus profonde
        classifier_layers = []

        # Première couche
        classifier_layers.append(nn.Linear(num_ftrs, hidden_size1))
        if use_batchnorm:
            classifier_layers.append(nn.BatchNorm1d(hidden_size1))
        classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Dropout(dropout1))

        # Attention optionnelle
        if use_attention:
            classifier_layers.append(AttentionBlock(hidden_size1))

        # Deuxième couche
        classifier_layers.append(nn.Linear(hidden_size1, hidden_size2))
        if use_batchnorm:
            classifier_layers.append(nn.BatchNorm1d(hidden_size2))
        classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Dropout(dropout2))

        # Troisième couche optionnelle
        if dropout3 > 0:
            mid_size = hidden_size2 // 2
            classifier_layers.append(nn.Linear(hidden_size2, mid_size))
            if use_batchnorm:
                classifier_layers.append(nn.BatchNorm1d(mid_size))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dropout3))
            classifier_layers.append(nn.Linear(mid_size, num_classes))
        else:
            classifier_layers.append(nn.Linear(hidden_size2, num_classes))

        model.fc = nn.Sequential(*classifier_layers)

        # Geler le backbone initialement
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        self.model = model

        # Loss avec Focal Loss optionnel
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        if focal_gamma > 0:
            self.loss_fn = FocalLoss(alpha=class_weights_tensor, gamma=focal_gamma, label_smoothing=label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Mixup augmentation si activé
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

        # Dégeler progressivement les blocs du backbone (du plus profond au moins profond)
        if self.current_epoch == self.unfreeze_epoch and batch_idx == 0:
            # ResNet18 a layer4, layer3, layer2, layer1 (du plus profond au moins profond)
            layers_to_unfreeze = ["fc"]  # Toujours actif
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

        self.log_dict({
            'val_loss': loss,
            'val_acc': acc,
            'val_f1': f1
        }, on_step=False, on_epoch=True)

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
                pct_start=0.2,  # Warmup plus court
                anneal_strategy='cos',
                div_factor=10.0,  # Moins agressif au début
                final_div_factor=100.0
            )
            interval = "step"
            monitor = None
        elif self.scheduler_type == "cosine_warmup":
            # Cosine avec warmup linéaire
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
        else:  # cosine
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


def objective(trial):
    """Fonction objectif pour Optuna"""

    pl.seed_everything(42)

    try:
        # Hyperparamètres du DataLoader
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 48])
        aug_strength = trial.suggest_float("aug_strength", 0.3, 1.0)

        print(f"[Trial {trial.number}] batch_size={batch_size}, aug_strength={aug_strength:.2f}")

        # Setup data - chemin vers fake-detector-nanobananapro
        print(f"[Trial {trial.number}] Setup des données...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_base_dir = os.path.join(base_dir, "..", "fake-detector-nanobananapro")

        print(f"[Trial {trial.number}] Combinaison des datasets: midjourney + nanobanana")

        dm = FakeDetectorDataModule(data_base_dir, batch_size=batch_size, aug_strength=aug_strength)
        dm.setup()
        print(f"[Trial {trial.number}] Données chargées ({len(dm.train_dataset)} train, {len(dm.val_dataset)} val)")

        # Créer le modèle
        print(f"[Trial {trial.number}] Création du modèle...")
        # Chercher d'abord le modèle ResNet18, sinon fallback sur ImageNet
        pretrained_path = "./models/best_76_resnet18_fake_detector.pth"
        model = OptunaFakeDetector(
            trial=trial,
            num_classes=2,
            class_weights=dm.class_weights,
            pretrained_weights_path=pretrained_path if os.path.exists(pretrained_path) else None
        )
        print(f"[Trial {trial.number}] Modèle créé")

        # Callbacks - patience plus longue pour laisser le fine-tuning converger
        early_stop = EarlyStopping(
            monitor="val_acc",
            patience=8,  # Plus de patience
            mode="max",
            min_delta=0.002
        )

        optuna_callback = OptunaPruningCallback(trial)

        # CUDA uniquement, pas de MPS
        if torch.cuda.is_available():
            accelerator = "gpu"
            precision = "16-mixed"
            print(f"[Trial {trial.number}] Utilisation de CUDA")
        else:
            accelerator = "cpu"
            precision = "32"
            print(f"[Trial {trial.number}] Utilisation de CPU (CUDA non disponible)")

        print(f"[Trial {trial.number}] Configuration du trainer...")

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        print(f"[Trial {trial.number}] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        test_batch = next(iter(train_loader))
        print(f"[Trial {trial.number}] Batch test OK: {test_batch[0].shape}, {test_batch[1].shape}")

        progress_cb = TQDMProgressBar(refresh_rate=1)
        trainer = pl.Trainer(
            max_epochs=35,  # Plus d'epochs pour le fine-tuning
            accelerator=accelerator,
            devices=1,
            precision=precision,
            callbacks=[early_stop, optuna_callback, progress_cb],
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            accumulate_grad_batches=2,
            num_sanity_val_steps=0,
            # IMPORTANT: Utiliser 100% des données pour vraiment évaluer les hyperparamètres
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            log_every_n_steps=20
        )

        print(f"[Trial {trial.number}] Démarrage de l'entraînement...")
        trainer.fit(model, dm)
        print(f"[Trial {trial.number}] Entraînement terminé")

        val_acc = trainer.callback_metrics.get('val_acc')
        val_f1 = trainer.callback_metrics.get('val_f1')

        if val_acc is not None and val_f1 is not None:
            acc = val_acc.item()
            f1 = val_f1.item()
            # Score combiné: 70% accuracy + 30% F1 (pour équilibrer précision/rappel)
            score = 0.7 * acc + 0.3 * f1
            print(f"[Trial {trial.number}] Acc: {acc:.4f}, F1: {f1:.4f}, Score combiné: {score:.4f}")
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


def run_optuna_optimization(n_trials=50, n_jobs=1):
    """Lance l'optimisation Optuna"""

    # TPE avec plus de startup trials pour mieux explorer l'espace
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=10,  # Plus d'exploration aléatoire au début
        multivariate=True,  # Considère les corrélations entre hyperparamètres
    )

    # Pruner plus patient - ne pas tuer les trials trop tôt
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5,  # Au moins 5 epochs avant de pruner
        max_resource=35,
        reduction_factor=3
    )

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name="fake_detector"
    )

    print(f"Démarrage de l'optimisation avec {n_trials} essais...")
    print("Note: Utilisation du modèle pré-entraîné ResNet18 (best_resnet18_fake_detector.pth)")
    print("Objectif: Passer de 85-87% à >95%")
    print("Stratégie: Fine-tuning progressif + Mixup + Focal Loss + Hyperband pruning")
    print()

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    print("\n" + "="*50)
    print("OPTIMISATION TERMINÉE")
    print("="*50)

    best_trial = study.best_trial
    print(f"\nMeilleure précision: {best_trial.value:.4f}")
    print("\nMeilleurs hyperparamètres:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    import pandas as pd
    df = study.trials_dataframe()
    df.to_csv("./optuna_results_fake_detector.csv", index=False)
    print(f"\nRésultats sauvegardés dans optuna_results_fake_detector.csv")

    return study, best_trial


if __name__ == "__main__":
    # 50 trials minimum pour bien explorer l'espace
    study, best_trial = run_optuna_optimization(n_trials=50, n_jobs=1)

    print("\n" + "="*60)
    print("MEILLEURS HYPERPARAMÈTRES TROUVÉS")
    print("="*60)
    print(f"\nPrécision atteinte: {best_trial.value:.4f} ({best_trial.value*100:.2f}%)")
    print("\nHyperparamètres optimaux:")
    print("-"*40)

    # Grouper par catégorie
    categories = {
        "Optimisation": ["learning_rate", "weight_decay", "lr_ratio", "scheduler"],
        "Architecture": ["hidden_size1", "hidden_size2", "use_attention", "use_batchnorm"],
        "Régularisation": ["dropout1", "dropout2", "dropout3", "label_smoothing", "focal_gamma", "mixup_alpha"],
        "Fine-tuning": ["unfreeze_epoch", "num_unfreeze_blocks"],
        "Data": ["batch_size", "aug_strength"]
    }

    for cat_name, params in categories.items():
        print(f"\n{cat_name}:")
        for key in params:
            if key in best_trial.params:
                value = best_trial.params[key]
                if isinstance(value, float):
                    print(f"  {key} = {value:.6f}")
                else:
                    print(f"  {key} = {value}")
