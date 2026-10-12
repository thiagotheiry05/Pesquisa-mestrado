import numpy as np

from metodologia_proposta.train import prepare_images


def get_patch_softmax_outputs_with_position(
    patches: np.ndarray,
    positions_norm: np.ndarray,
    model,
    batch_size: int = 256,
):
    """
    Retorna as probabilidades Softmax de cada patch.

    Entrada:
        patches: (N, 64, 64)
        positions_norm: (N, 2)

    Saída:
        probabilities: (N, num_classes)
    """

    X_img = prepare_images(patches)
    X_pos = positions_norm.astype(np.float32)

    probabilities = model.predict(
        [X_img, X_pos],
        batch_size=batch_size,
        verbose=1,
    )

    return probabilities


def reconstruct_softmax_images(
    patch_softmax_outputs: np.ndarray,
    positions_pixel: np.ndarray,
    image_shape: tuple,
    patch_size: int,
):
    """
    Reconstrói um mapa de probabilidade para cada classe.
    """

    height, width = image_shape[:2]
    num_classes = patch_softmax_outputs.shape[1]

    softmax_images = np.zeros(
        (num_classes, height, width),
        dtype=np.float32,
    )

    for probabilities, (row, col) in zip(
        patch_softmax_outputs,
        positions_pixel,
    ):
        end_row = min(row + patch_size, height)
        end_col = min(col + patch_size, width)

        softmax_images[
            :,
            row:end_row,
            col:end_col,
        ] = probabilities[:, np.newaxis, np.newaxis]

    return softmax_images


def reconstruct_confidence_image(
    patch_softmax_outputs: np.ndarray,
    positions_pixel: np.ndarray,
    image_shape: tuple,
    patch_size: int,
):
    """
    Reconstrói o mapa da maior probabilidade Softmax
    de cada patch.
    """

    confidence_values = np.max(
        patch_softmax_outputs,
        axis=1,
    )

    height, width = image_shape[:2]

    confidence_image = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    for confidence, (row, col) in zip(
        confidence_values,
        positions_pixel,
    ):
        end_row = min(row + patch_size, height)
        end_col = min(col + patch_size, width)

        confidence_image[
            row:end_row,
            col:end_col,
        ] = confidence

    return confidence_image