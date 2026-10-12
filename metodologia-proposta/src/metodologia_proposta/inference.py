import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from metodologia_proposta.config import (
    PATCH_SIZE,
    CLASS_NAMES,
    COLOR_TO_CLASS,
    RECONSTRUCTION_BATCH_SIZE,
)
from metodologia_proposta.reconstruction import (
    divide_image_into_patches_with_positions,
    reconstruct_image_from_patches,
)
from metodologia_proposta.softmax import (
    get_patch_softmax_outputs_with_position,
    reconstruct_softmax_images,
    reconstruct_confidence_image,
)
from metodologia_proposta.visualization import (
    plot_softmax_images,
    plot_softmax_combined,
)

def predict_seismic_line(
    model,
    image_path: Path,
    output_dir: Path,
):
    """
    Executa a segmentação e gera os mapas Softmax
    para uma seção sísmica não vista.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar a nova seção
    print("\nCarregando seção sísmica não vista...")

    image = np.load(image_path)

    print(f"Shape da imagem: {image.shape}")

    # 2. Dividir em patches
    patches, positions_pixel, positions_norm = (
        divide_image_into_patches_with_positions(
            image=image,
            patch_size=PATCH_SIZE,
        )
    )

    print(f"Quantidade de patches: {len(patches)}")

    # 3. Obter probabilidades Softmax
    print("\nExecutando predições...")

    patch_softmax_outputs = get_patch_softmax_outputs_with_position(
        patches=patches,
        positions_norm=positions_norm,
        model=model,
        batch_size=RECONSTRUCTION_BATCH_SIZE,
    )

    # 4. Classificar os patches
    class_indices = np.argmax(
        patch_softmax_outputs,
        axis=1,
    )

    class_to_color = {
        class_name: color
        for color, class_name in COLOR_TO_CLASS.items()
    }

    colors = np.array(
        [
            class_to_color[CLASS_NAMES[index]]
            for index in class_indices
        ],
        dtype=np.uint8,
    )

    colored_patches = np.broadcast_to(
        colors[:, np.newaxis, np.newaxis, :],
        (
            len(patches),
            PATCH_SIZE,
            PATCH_SIZE,
            3,
        ),
    )

    # 5. Reconstruir a segmentação
    segmented_image = reconstruct_image_from_patches(
        patches=colored_patches,
        positions_pixel=positions_pixel,
        image_shape=image.shape,
    )

    plt.imsave(
        output_dir / "segmented_image.png",
        segmented_image,
    )

    print("Imagem segmentada salva.")

    # 6. Reconstruir mapas Softmax
    softmax_images = reconstruct_softmax_images(
        patch_softmax_outputs=patch_softmax_outputs,
        positions_pixel=positions_pixel,
        image_shape=image.shape,
        patch_size=PATCH_SIZE,
    )

    # 7. Reconstruir confiança
    confidence_image = reconstruct_confidence_image(
        patch_softmax_outputs=patch_softmax_outputs,
        positions_pixel=positions_pixel,
        image_shape=image.shape,
        patch_size=PATCH_SIZE,
    )

    # 8. Salvar mapas individuais
    plot_softmax_images(
        softmax_images=softmax_images,
        confidence_image=confidence_image,
        class_names=CLASS_NAMES,
        output_dir=output_dir,
    )

    # 9. Salvar imagem combinada
    plot_softmax_combined(
        softmax_images=softmax_images,
        class_names=CLASS_NAMES,
        output_dir=output_dir,
    )

    print("\nInferência concluída.")
    print(f"Resultados salvos em: {output_dir}")