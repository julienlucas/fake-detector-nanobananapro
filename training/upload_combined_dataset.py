#!/usr/bin/env python3
"""
Script pour combiner deux datasets d'images et les uploader sur HuggingFace.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image
from huggingface_hub import login

# Charger le .env depuis le répertoire racine du projet
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Configuration
BASE_PATH = Path("/Users/Julien/Desktop/fake-detector-nanobananapro")
DATASETS = [
    BASE_PATH / "AIvsReal_midjourney_dalle_sd",
    BASE_PATH / "AIvsReal_nanobanana_pro",
]
HF_REPO = "julienlucas/test"


def collect_images(datasets_paths: list[Path], split: str) -> tuple[list[str], list[int]]:
    """Collecte les chemins d'images et labels pour un split donné."""
    image_paths = []
    labels = []

    for dataset_path in datasets_paths:
        split_path = dataset_path / split

        # fake = 0, real = 1
        for label_name, label_id in [("fake", 0), ("real", 1)]:
            label_path = split_path / label_name
            if not label_path.exists():
                print(f"Warning: {label_path} n'existe pas")
                continue

            for img_file in label_path.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    image_paths.append(str(img_file))
                    labels.append(label_id)

    return image_paths, labels


def create_dataset(image_paths: list[str], labels: list[int]) -> Dataset:
    """Crée un Dataset HuggingFace à partir des chemins d'images."""
    return Dataset.from_dict({
        "image": image_paths,
        "label": labels,
    }).cast_column("image", Image())


def main():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_ACCESS_TOKEN")

    if not token:
        print("Token HuggingFace non trouvé. Définissez HF_ACCESS_TOKEN.")
        return

    login(token=token.strip())

    print("Collecte des images...")

    # Collecter train et test
    train_paths, train_labels = collect_images(DATASETS, "train")
    test_paths, test_labels = collect_images(DATASETS, "test")

    print(f"Train: {len(train_paths)} images (fake: {train_labels.count(0)}, real: {train_labels.count(1)})")
    print(f"Test: {len(test_paths)} images (fake: {test_labels.count(0)}, real: {test_labels.count(1)})")

    print("\nCréation du dataset HuggingFace...")

    train_dataset = create_dataset(train_paths, train_labels)
    test_dataset = create_dataset(test_paths, test_labels)

    # Ajouter les features avec ClassLabel
    features = Features({
        "image": Image(),
        "label": ClassLabel(names=["fake", "real"])
    })

    train_dataset = train_dataset.cast(features)
    test_dataset = test_dataset.cast(features)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    print(f"\nUpload sur {HF_REPO}...")
    dataset_dict.push_to_hub(
        HF_REPO,
        private=True,
        token=token.strip()
    )

    print(f"\nDataset uploadé sur https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
