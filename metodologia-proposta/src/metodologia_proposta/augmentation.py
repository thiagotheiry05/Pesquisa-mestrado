import albumentations as A
import numpy as np
import cv2

transform_basic = A.Compose(
    [
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.08,
        rotate_limit=0,
        border_mode=cv2.BORDER_REFLECT,
        p=0.6
    ),

    A.RandomCrop(
        height=56,
        width=56,
        p=0.3
    ),

    A.PadIfNeeded(
        min_height=64,
        min_width=64,
        border_mode=cv2.BORDER_REFLECT,
        p=1.0
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.6
    ),

    # A.GaussNoise(
    #     p=0.4
    # ),

    A.Blur(
        blur_limit=3,
        p=0.2
    ),
    ]
)


def augment_training_data(
    images: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    target_per_class: int,
    random_seed: int = 42,
):
    """
    Aplica data augmentation somente ao conjunto de treinamento,
    buscando atingir target_per_class exemplos para cada classe.

    Os exemplos originais são mantidos e novos patches são
    gerados a partir deles.
    """

    rng = np.random.default_rng(random_seed)

    augmented_images = []
    augmented_positions = []
    augmented_labels = []

    classes = np.unique(labels)

    for class_index in classes:
        class_indices = np.where(labels == class_index)[0]

        current_count = len(class_indices)
        samples_needed = max(0, target_per_class - current_count)

        # Mantém todos os exemplos originais.
        augmented_images.extend(images[class_indices])
        augmented_positions.extend(positions[class_indices])
        augmented_labels.extend(labels[class_indices])

        # Gera novos exemplos somente se necessário.
        for _ in range(samples_needed):
            selected_index = rng.choice(class_indices)

            image = images[selected_index]
            position = positions[selected_index]

            transformed = transform_basic(image=image)

            augmented_image = transformed["image"]

            augmented_images.append(augmented_image)
            augmented_positions.append(position)
            augmented_labels.append(class_index)

    return (
        np.asarray(augmented_images),
        np.asarray(augmented_positions, dtype=np.float32),
        np.asarray(augmented_labels, dtype=np.int64),
    )