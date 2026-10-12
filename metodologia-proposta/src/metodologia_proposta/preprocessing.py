import numpy as np
from sklearn.model_selection import train_test_split
from metodologia_proposta.config import CLASS_NAMES, RANDOM_SEED


CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


def encode_labels(labels: np.ndarray) -> np.ndarray:
    """Converte os nomes das classes para índices numéricos."""
    return np.asarray(
        [CLASS_TO_INDEX[label] for label in labels],
        dtype=np.int64,
    )


def split_dataset(
    image_patches: np.ndarray,
    positions_norm: np.ndarray,
    positions_pixel: np.ndarray,
    labels: np.ndarray,
):
    """
    Divide os dados em:
        70% treino
        15% validação
        15% teste

    A divisão é estratificada pelas classes e reproduz
    a estratégia utilizada no notebook original.
    """

    # Primeira divisão:
    # 70% treino e 30% temporário.
    (
        X_img_train,
        X_img_temp,
        X_pos_train,
        X_pos_temp,
        X_pos_pixel_train,
        X_pos_pixel_temp,
        y_train,
        y_temp,
    ) = train_test_split(
        image_patches,
        positions_norm,
        positions_pixel,
        labels,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    # Segunda divisão:
    # divide os 30% restantes igualmente:
    # 15% validação e 15% teste.
    (
        X_img_val,
        X_img_test,
        X_pos_val,
        X_pos_test,
        X_pos_pixel_val,
        X_pos_pixel_test,
        y_val,
        y_test,
    ) = train_test_split(
        X_img_temp,
        X_pos_temp,
        X_pos_pixel_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    return {
        "X_img_train": X_img_train,
        "X_pos_train": X_pos_train,
        "X_pos_pixel_train": X_pos_pixel_train,
        "y_train": y_train,

        "X_img_val": X_img_val,
        "X_pos_val": X_pos_val,
        "X_pos_pixel_val": X_pos_pixel_val,
        "y_val": y_val,

        "X_img_test": X_img_test,
        "X_pos_test": X_pos_test,
        "X_pos_pixel_test": X_pos_pixel_test,
        "y_test": y_test,
    }