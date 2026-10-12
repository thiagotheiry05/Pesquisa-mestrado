import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns


def plot_augmentation_examples(
    original_images: np.ndarray,
    augmented_images: np.ndarray,
    num_examples: int = 5,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Compara patches originais e aumentados utilizando
    a mesma escala de amplitudes.
    """

    num_examples = min(
        num_examples,
        len(original_images),
        len(augmented_images),
    )

    if vmin is None:
        vmin = np.percentile(original_images, 1)

    if vmax is None:
        vmax = np.percentile(original_images, 99)

    fig, axes = plt.subplots(
        2,
        num_examples,
        figsize=(3 * num_examples, 6),
    )

    for i in range(num_examples):
        axes[0, i].imshow(
            original_images[i],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(
            augmented_images[i],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        axes[1, i].set_title("Augmented")
        axes[1, i].axis("off")

        print(f"vmin: {vmin:.4f}")
        print(f"vmax: {vmax:.4f}")

    plt.tight_layout()
    plt.show()

def plot_training_history(history, output_dir: Path):
    """
    Salva os gráficos de loss e accuracy.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    plt.figure(figsize=(8, 5))

    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")

    plt.title("Perda durante o treinamento")
    plt.xlabel("Épocas")
    plt.ylabel("Perda")
    plt.legend()

    plt.savefig(
        output_dir / "training_loss.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    # Accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(history.history["accuracy"], label="Treino")
    plt.plot(history.history["val_accuracy"], label="Validação")

    plt.title("Acurácia durante o treinamento")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()

    plt.savefig(
        output_dir / "training_accuracy.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()


def plot_confusion_matrix(
    conf_matrix,
    class_names,
    output_dir: Path,
):
    """
    Salva a matriz de confusão.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")

    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)

    plt.savefig(
        output_dir / "confusion_matrix.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

def plot_reconstruction(
    segmented_image: np.ndarray,
    reference_mask: np.ndarray,
    output_dir: Path,
):
    """
    Salva a reconstrução e a máscara original
    em imagens separadas.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.imsave(
        output_dir / "segmented_image.png",
        segmented_image,
    )

    plt.imsave(
        output_dir / "reference_mask.png",
        reference_mask,
    )

    print("\nImagens salvas:")
    print(output_dir / "segmented_image.png")
    print(output_dir / "reference_mask.png")


def plot_softmax_images(
    softmax_images: np.ndarray,
    confidence_image: np.ndarray,
    class_names: list,
    output_dir: Path,
):
    """
    Salva os mapas de probabilidade por classe
    e o mapa de confiança máxima.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    for index, class_name in enumerate(class_names):
        fig, ax = plt.subplots(figsize=(14, 5))

        im = ax.imshow(
            softmax_images[index],
            cmap="viridis",
            vmin=0,
            vmax=1,
            aspect="auto",
        )

        ax.set_title(f"Softmax - {class_name}")
        ax.axis("off")

        fig.colorbar(
            im,
            ax=ax,
            label="Probabilidade Softmax",
        )

        filename = class_name.replace(" ", "_")

        fig.savefig(
            output_dir / f"softmax_{filename}.png",
            bbox_inches="tight",
            dpi=200,
        )

        plt.close(fig)

    # Mapa de confiança
    fig, ax = plt.subplots(figsize=(14, 5))

    im = ax.imshow(
        confidence_image,
        cmap="viridis",
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_title("Confiança máxima do Softmax")
    ax.axis("off")

    fig.colorbar(
        im,
        ax=ax,
        label="Confiança Softmax",
    )

    fig.savefig(
        output_dir / "softmax_confidence.png",
        bbox_inches="tight",
        dpi=200,
    )

    plt.close(fig)

def plot_softmax_combined(
    softmax_images: np.ndarray,
    class_names: list,
    output_dir: Path,
):
    """
    Salva uma única figura com os mapas Softmax
    das cinco classes, empilhados verticalmente.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Nomes exibidos na figura
    display_names = {
        "agua": "Water",
        "bacia sedimentar": "Sedimentary basin",
        "crosta continental": "Continental crust",
        "crosta oceanica": "Oceanic crust",
        "manto": "Mantle",
    }

    num_classes = len(class_names)

    fig, axes = plt.subplots(
        num_classes,
        1,
        figsize=(14, 9),
        sharex=True,
        constrained_layout=True,
    )

    for index, (ax, class_name) in enumerate(
        zip(axes, class_names)
    ):
        im = ax.imshow(
            softmax_images[index],
            cmap="viridis",
            vmin=0,
            vmax=1,
            aspect="auto",
        )

        ax.set_title(
            display_names.get(class_name, class_name),
            fontsize=12,
        )

        ax.axis("off")

    # Uma única barra de cores para todas as classes
    colorbar = fig.colorbar(
        im,
        ax=axes,
        fraction=0.025,
        pad=0.015,
    )

    colorbar.set_label(
        "Softmax Probability",
        fontsize=12,
    )

    fig.savefig(
        output_dir / "softmax_combined.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    plt.close(fig)

    print(
        f"Figura Softmax combinada salva em: "
        f"{output_dir / 'softmax_combined.png'}"
    )