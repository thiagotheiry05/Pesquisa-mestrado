from pathlib import Path

import cv2
import numpy as np


def load_seismic_data(file_path: Path) -> np.ndarray:
    """Carrega os dados sísmicos armazenados em um arquivo .npy."""
    return np.load(file_path)


def load_mask(file_path: Path) -> np.ndarray:
    """Carrega uma máscara de imagem e converte de BGR para RGB."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo da máscara não encontrado: {file_path}"
        )

    image_bytes = np.fromfile(file_path, dtype=np.uint8)
    mask = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if mask is None:
        raise ValueError(
            f"Não foi possível decodificar a máscara: {file_path}"
        )

    return cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)