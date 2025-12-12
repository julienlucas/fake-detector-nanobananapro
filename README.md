# Fake Image Finder - Nano Banana Pro

Détecteur d'images générées par IA utilisant un modèle MobileNetV3 fine-tuné pour identifier spécifiquement les images créées par **Nano Banana Pro** (modèle d'IA multimodale de Google).

## 🎯 Objectif

Ce projet vise à distinguer les images **réelles** des images **générées par IA**, avec un focus particulier sur la détection des images créées par Nano Banana Pro. Le modèle est entraîné en deux étapes :

1. **Entraînement initial** : Détection générale d'images fake (Stable Diffusion, Midjourney, DALL-E)
2. **Fine-tuning** : Adaptation spécifique pour détecter les images Nano Banana Pro

## 🏗️ Architecture

- **Modèle de base** : MobileNetV3-Large
- **Pré-entraînement** : ImageNet
- **Fine-tuning** : Classifier uniquement (features gelées)
- **Classes** : 2 (Real / Fake)
- **Résolution d'entrée** : 224x224

## 📁 Structure du Projet

```
fake_image_finder/
├── AIvsReal_sampled/          # Dataset initial (SD, Midjourney, DALL-E)
│   ├── train/
│   │   ├── fake/
│   │   └── real/
│   └── test/
│       ├── fake/
│       └── real/
├── AIvsReal_nanobanana_pro/   # Dataset Nano Banana Pro
│   ├── train/
│   │   ├── fake/              # 2250 images
│   │   └── real/
│   └── test/
│       ├── fake/              # 500 images
│       └── real/
├── train.py                    # Entraînement initial
├── finetune_nanobananapro.py  # Fine-tuning Nano Banana Pro
├── inference.py                # Inférence avec Grad-CAM
├── inference_check_fulldataset.py  # Évaluation complète du dataset
├── best_model.pth              # Modèle initial (SD/Midjourney/DALL-E)
├── best_model_nanobanana.pth   # Modèle fine-tuné Nano Banana Pro
└── utils/
    ├── training.py             # Boucle d'entraînement
    ├── helper_utils.py         # Utilitaires
    └── visualization.py        # Visualisation Grad-CAM
```

## 🚀 Utilisation

### Installation

```bash
# Installation des dépendances avec uv
uv sync
```

### Entraînement

#### 1. Entraînement initial (SD, Midjourney, DALL-E)

```bash
python train.py
```

Génère `best_model.pth` - modèle de base pour détecter les images fake générales.

#### 2. Fine-tuning pour Nano Banana Pro

```bash
python finetune_nanobananapro.py
```

Génère `best_model_nanobanana.pth` - modèle adapté pour Nano Banana Pro.

**Configuration du fine-tuning :**
- Learning rate : 0.0005
- Batch size : 32
- Epochs : 1 (convergence rapide)
- Data augmentation : RandomResizedCrop, flips, rotations, color jitter, perspective

### Inférence

#### Inférence simple avec visualisation Grad-CAM

```bash
python inference.py
```

Affiche la prédiction et les régions importantes de l'image.

#### Évaluation complète du dataset de test

```bash
python inference_check_fulldataset.py
```

Teste toutes les images du dataset `test/real` et `test/fake` et affiche :
- Précision, Recall, F1-Score par classe
- Accuracy globale
- Statistiques détaillées

## 📊 Performances

### Modèle fine-tuné Nano Banana Pro

- **Accuracy globale** : ~89-90%
- **Précision REAL** : ~89%
- **Recall REAL** : ~89%
- **Précision FAKE** : ~89%
- **Recall FAKE** : ~89%

### Dataset

- **Train** : 2250 images fake Nano Banana Pro + images real
- **Test** : 500 images fake Nano Banana Pro + images real
- **Ratio** : ~82% train / 18% test

## 📥 Sources des Images Nano Banana Pro

Les images Nano Banana Pro utilisées pour l'entraînement ont été collectées depuis :

- **[YouMind](https://youmind.com/fr-FR/nano-banana-pro-prompts)** - Collection de prompts et images Nano Banana Pro
- **[Higgsfield.ai](https://higgsfield.ai/nano-banana-pro-preview)** - Aperçu et exemples Nano Banana Pro
- **[Awesome Nano Banana Pro (GitHub)](https://github.com/ZeroLu/awesome-nanobanana-pro)** - Collection open-source d'exemples
- **[PromptGather.io](https://promptgather.io)** - Plateforme de collecte de prompts Nano Banana Pro
- **[Google Sheets - PromptGather](https://docs.google.com/spreadsheets/d/1GAp_yaqAX9y_K8lnGQw9pe_BTpHZehoonaxi4whEQIE/edit?gid=116507383#gid=116507383)** - Base de données de prompts avec images

## 🔧 Configuration

### Transformations d'entraînement

- `RandomResizedCrop(224, 224)` - scale (0.7, 1.0)
- `RandomHorizontalFlip` - p=0.5
- `RandomVerticalFlip` - p=0.2
- `RandomRotation` - degrees=20
- `ColorJitter` - brightness, contrast, saturation, hue
- `RandomAffine` - translate, scale
- `RandomPerspective` - p=0.3

### Transformations de validation

- `Resize(256, 256)`
- `CenterCrop(224)`
- Normalisation ImageNet

## 📝 Notes Techniques

- **Device** : MPS (Apple Silicon) ou CPU
- **Framework** : PyTorch
- **Optimiseur** : Adam (lr=0.0005)
- **Loss** : CrossEntropyLoss
- **Seuils de confiance** : 0.7 pour REAL et FAKE

## 🎨 Fonctionnalités

- ✅ Détection d'images fake/real
- ✅ Visualisation Grad-CAM pour comprendre les décisions
- ✅ Fine-tuning spécifique Nano Banana Pro
- ✅ Évaluation complète avec métriques détaillées
- ✅ Support des formats : JPG, PNG, WebP

## 📄 Licence

Ce projet est destiné à la recherche et à l'éducation sur la détection d'images générées par IA.
