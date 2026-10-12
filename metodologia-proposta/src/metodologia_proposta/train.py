import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model


def prepare_images(images: np.ndarray) -> np.ndarray:
    """
    Prepara os patches para a entrada da CNN.

    Entrada:
        (N, 64, 64)

    Saída:
        (N, 64, 64, 1)
    """

    images = images.astype(np.float32)

    if images.ndim == 3:
        images = images[..., np.newaxis]

    return images


def train_model(
    model: Model,
    X_img_train: np.ndarray,
    X_pos_train: np.ndarray,
    y_train: np.ndarray,
    X_img_val: np.ndarray,
    X_pos_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    model_path,
    early_stopping_monitor: str = "val_loss",
    early_stopping_patience: int = 5,
):
    X_img_train = prepare_images(X_img_train)
    X_img_val = prepare_images(X_img_val)

    print("\nDados enviados ao modelo:")

    print("Treino:")
    print(f"  Imagens: {X_img_train.shape}")
    print(f"  Posições: {X_pos_train.shape}")
    print(f"  Labels: {y_train.shape}")

    print("\nValidação:")
    print(f"  Imagens: {X_img_val.shape}")
    print(f"  Posições: {X_pos_val.shape}")
    print(f"  Labels: {y_val.shape}")

    early_stopping = EarlyStopping(
        monitor=early_stopping_monitor,
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor=early_stopping_monitor,
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        x=[
            X_img_train,
            X_pos_train,
        ],
        y=y_train,
        validation_data=(
            [
                X_img_val,
                X_pos_val,
            ],
            y_val,
        ),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[
            early_stopping,
            model_checkpoint,
        ],
    )

    return history