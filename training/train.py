import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
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
    def __init__(self, data_base_dir, batch_size=32, aug_strength=0.6):
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


class FocalLoss(nn.Module):
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


class FakeDetectorClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        learning_rate=3.62e-5,
        weight_decay=1.68e-5,
        lr_ratio=0.132,
        dropout1=0.351,
        dropout2=0.293,
        focal_gamma=0.805,
        mixup_alpha=0.138,
        label_smoothing=0.05,
        scheduler_type="cosine",
        unfreeze_epoch=3,
        num_unfreeze_blocks=2,
        batch_size_factor=2,
        class_weights=None,
        pretrained_weights_path=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_ratio = lr_ratio
        self.label_smoothing = label_smoothing
        self.scheduler_type = scheduler_type
        self.focal_gamma = focal_gamma
        self.mixup_alpha = mixup_alpha
        self.unfreeze_epoch = unfreeze_epoch
        self.num_unfreeze_blocks = num_unfreeze_blocks
        self.batch_size_factor = batch_size_factor

        model = tv_models.resnet18(weights=None)
        num_ftrs = model.fc.in_features

        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            state_dict = torch.load(pretrained_weights_path, map_location='cpu', weights_only=True)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model_state = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items()
                           if k in model_state and model_state[k].shape == v.shape}
            model.load_state_dict(filtered_dict, strict=False)
            print("Backbone pré-entraîné chargé")
        else:
            model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

        hidden_size1 = 512
        hidden_size2 = 256
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

        self.log_dict({
            'val_loss': loss,
            'val_acc': acc,
            'val_f1': f1
        }, on_step=False, on_epoch=True, prog_bar=True)

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

        if self.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7
            )
            interval = "epoch"
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
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7
            )
            interval = "epoch"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": interval}
        }


class ChestXRayDataModule:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        train_transform, val_transform = define_transformations()
        train_path = os.path.join(self.data_dir, "train")
        val_path = os.path.join(self.data_dir, "test")

        if os.path.exists(train_path):
            self.train_dataset = ImageFolder(root=train_path, transform=train_transform)

        if not os.path.exists(val_path):
            abs_val_path = os.path.abspath(val_path)
            abs_data_dir = os.path.abspath(self.data_dir)
            raise FileNotFoundError(
                f"Répertoire de validation introuvable: {val_path}\n"
                f"Chemin absolu: {abs_val_path}\n"
                f"Répertoire parent: {abs_data_dir}\n"
                f"Assurez-vous que le répertoire 'test' existe dans: {abs_data_dir}"
            )

        self.val_dataset = ImageFolder(root=val_path, transform=val_transform)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)


def define_transformations(mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return None, val_transform


if __name__ == "__main__":
    pl.seed_everything(42)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_base_dir = os.path.join(base_dir, "..", "fake-detector-nanobananapro")
    pretrained_path = "./models/best_76_resnet18_fake_detector.pth"

    batch_size = 32
    aug_strength = 0.6

    dm = FakeDetectorDataModule(data_base_dir, batch_size=batch_size, aug_strength=aug_strength)
    dm.setup()

    model = FakeDetectorClassifier(
        num_classes=2,
        learning_rate=3.62e-5,
        weight_decay=1.68e-5,
        lr_ratio=0.132,
        dropout1=0.351,
        dropout2=0.293,
        focal_gamma=0.805,
        mixup_alpha=0.138,
        label_smoothing=0.05,
        scheduler_type="cosine",
        unfreeze_epoch=3,
        num_unfreeze_blocks=2,
        batch_size_factor=2,
        class_weights=dm.class_weights,
        pretrained_weights_path=pretrained_path if os.path.exists(pretrained_path) else None
    )

    early_stop = EarlyStopping(
        monitor="val_acc",
        patience=8,
        mode="max",
        min_delta=0.005
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
        precision = "16-mixed"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        precision = "32"
    else:
        accelerator = "cpu"
        precision = "32"

    progress_cb = TQDMProgressBar(refresh_rate=1)
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=[early_stop, progress_cb],
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        gradient_clip_val=1.0,
        accumulate_grad_batches=model.batch_size_factor,
        num_sanity_val_steps=0,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        log_every_n_steps=20
    )

    print("Démarrage de l'entraînement avec hyperparamètres optimaux d'Optuna...")
    trainer.fit(model, dm)

    os.makedirs("./models", exist_ok=True)
    torch.save(model.model.state_dict(), './models/best_resnet18_fake_detector_optuna.pth')
    print("Modèle sauvegardé : models/best_resnet18_fake_detector_optuna.pth")
