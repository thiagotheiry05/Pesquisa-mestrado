import numpy as np

from metodologia_proposta.train import prepare_images


def divide_image_into_patches_with_positions(
    image: np.ndarray,
    patch_size: int,
):
    """
    Divide a imagem em patches, preservando as bordas
    por meio de preenchimento com zeros.
    """

    patches = []
    positions_pixel = []
    positions_norm = []

    height, width = image.shape[:2]

    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):

            patch = image[
                row:row + patch_size,
                col:col + patch_size,
            ]

            padded_patch = np.zeros(
                (patch_size, patch_size, *image.shape[2:]),
                dtype=image.dtype,
            )

            padded_patch[
                :patch.shape[0],
                :patch.shape[1],
            ] = patch

            patches.append(padded_patch)

            positions_pixel.append((row, col))

            center_x = col + patch_size // 2
            center_y = row + patch_size // 2

            positions_norm.append(
                (
                    center_x / width,
                    center_y / height,
                )
            )

    return (
        np.asarray(patches),
        np.asarray(positions_pixel),
        np.asarray(positions_norm, dtype=np.float32),
    )


def classify_and_color_patches(
    patches: np.ndarray,
    positions_norm: np.ndarray,
    model,
    class_to_color: dict,
    class_names: list,
    batch_size: int = 256,
):
    """
    Classifica os patches e atribui uma cor
    a cada patch conforme a classe prevista.
    """

    patches_in = prepare_images(patches)

    probabilities = model.predict(
        [patches_in, positions_norm],
        batch_size=batch_size,
        verbose=1,
    )

    class_indices = np.argmax(probabilities, axis=1)

    colors = np.asarray(
        [
            class_to_color[class_names[index]]
            for index in class_indices
        ],
        dtype=np.uint8,
    )

    colored_patches = np.broadcast_to(
        colors[:, np.newaxis, np.newaxis, :],
        (
            len(patches),
            patches.shape[1],
            patches.shape[2],
            3,
        ),
    )

    return colored_patches


def reconstruct_image_from_patches(
    patches: np.ndarray,
    positions_pixel: np.ndarray,
    image_shape: tuple,
):
    """
    Reconstroi a imagem colorida usando as
    posições originais dos patches.
    """

    reconstructed_image = np.zeros(
        (image_shape[0], image_shape[1], 3),
        dtype=np.uint8,
    )

    for patch, (row, col) in zip(
        patches,
        positions_pixel,
    ):
        patch_height, patch_width = patch.shape[:2]

        end_row = min(
            row + patch_height,
            image_shape[0],
        )

        end_col = min(
            col + patch_width,
            image_shape[1],
        )

        reconstructed_image[
            row:end_row,
            col:end_col,
        ] = patch[
            :end_row - row,
            :end_col - col,
        ]

    return reconstructed_image