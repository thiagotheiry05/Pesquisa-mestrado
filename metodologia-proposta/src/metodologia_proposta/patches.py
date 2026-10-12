import numpy as np


def get_dominant_class(
    mask_patch: np.ndarray,
    color_to_class: dict,
) -> str:
    """
    Retorna a classe correspondente à cor predominante
    em um patch da máscara.
    """
    reshaped_patch = mask_patch.reshape(-1, 3)

    unique_colors, counts = np.unique(
        reshaped_patch,
        axis=0,
        return_counts=True,
    )

    dominant_color = unique_colors[np.argmax(counts)]

    return color_to_class.get(
        tuple(dominant_color),
        "unknown",
    )


def create_training_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    color_to_class: dict,
):
    """
    Divide a imagem sísmica e sua máscara em patches.

    A imagem e a máscara são inicialmente recortadas para
    dimensões múltiplas de patch_size, reproduzindo o
    comportamento utilizado no notebook original.

    Retorna:
        image_patches
        mask_patches
        patch_labels
        positions_pixel
        positions_norm
    """

    height, width = image.shape[:2]

    # Ajusta as dimensões para múltiplos do tamanho do patch.
    new_height = (height // patch_size) * patch_size
    new_width = (width // patch_size) * patch_size

    adjusted_image = image[:new_height, :new_width]
    adjusted_mask = mask[:new_height, :new_width]

    image_patches = []
    mask_patches = []
    positions_pixel = []
    positions_norm = []

    for row in range(0, new_height, patch_size):
        for col in range(0, new_width, patch_size):

            image_patch = adjusted_image[
                row : row + patch_size,
                col : col + patch_size,
            ]

            mask_patch = adjusted_mask[
                row : row + patch_size,
                col : col + patch_size,
                :,
            ]

            image_patches.append(image_patch)
            mask_patches.append(mask_patch)

            # Posição original em pixels.
            positions_pixel.append((row, col))

            # Posição do centro do patch normalizada.
            center_x = col + patch_size // 2
            center_y = row + patch_size // 2

            positions_norm.append(
                (
                    center_x / new_width,
                    center_y / new_height,
                )
            )

    patch_labels = [
        get_dominant_class(mask_patch, color_to_class)
        for mask_patch in mask_patches
    ]

    return (
        np.asarray(image_patches),
        np.asarray(mask_patches),
        np.asarray(patch_labels),
        np.asarray(positions_pixel),
        np.asarray(positions_norm, dtype=np.float32),
    )