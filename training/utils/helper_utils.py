import os

import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png", show=True):
    fig = plt.figure(figsize=(8, 6))
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
    plt.tight_layout()

    abs_path = os.path.abspath(os.path.expanduser(save_path))
    fig.savefig(abs_path, dpi=150)
    print(f"Matrice de confusion enregistrée: {abs_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def plot_prediction_distribution(fake_count, real_count, show=True):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["fake", "real"], [fake_count, real_count], color=["#f44336", "#4CAF50"])
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution des prédictions (sans labels)")
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)
