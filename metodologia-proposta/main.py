from collections import Counter
import numpy as np
from metodologia_proposta.model import create_model

from metodologia_proposta.config import (AUGMENTATION_TARGET_PER_CLASS,CLASS_NAMES, COLOR_TO_CLASS,PATCH_SIZE,RANDOM_SEED, RAW_DATA_DIR,
                                         D_MODEL, MLP_DIM, NUM_HEADS, NUM_TRANSFORMER_LAYERS, TRANSFORMER_DROPOUT, BATCH_SIZE,
                                         EPOCHS, EARLY_STOPPING_MONITOR, EARLY_STOPPING_PATIENCE,MODELS_DIR, OUTPUTS_DIR, TRAIN_MODEL,
                                         RECONSTRUCTION_BATCH_SIZE, UNSEEN_IMAGE_PATH)
from metodologia_proposta.data import load_mask,load_seismic_data
from metodologia_proposta.patches import create_training_patches

from metodologia_proposta.preprocessing import encode_labels,split_dataset

from metodologia_proposta.augmentation import augment_training_data, transform_basic

from metodologia_proposta.visualization import plot_augmentation_examples
from metodologia_proposta.train import train_model

from tensorflow.keras.models import load_model

from metodologia_proposta.evaluation import evaluate_model

from metodologia_proposta.visualization import ( plot_softmax_combined, plot_training_history, plot_confusion_matrix,plot_reconstruction, plot_softmax_images)

from metodologia_proposta.reconstruction import (
    divide_image_into_patches_with_positions,
    classify_and_color_patches,
    reconstruct_image_from_patches,
)

from metodologia_proposta.softmax import (
    get_patch_softmax_outputs_with_position,
    reconstruct_softmax_images,
    reconstruct_confidence_image,
)

from metodologia_proposta.inference import predict_seismic_line

def main():
    seismic_path = RAW_DATA_DIR / "BB08RE11-210-PRCMIG.npy"
    mask_path = RAW_DATA_DIR / "Mascara_210_4.png"

    seismic_data = load_seismic_data(seismic_path)
    mask = load_mask(mask_path)

    print("Dados carregados:")
    print(f"  Sísmico: {seismic_data.shape}")
    print(f"  Máscara: {mask.shape}")

    (
        image_patches,
        mask_patches,
        patch_labels,
        positions_pixel,
        positions_norm,
    ) = create_training_patches(
        image=seismic_data,
        mask=mask,
        patch_size=PATCH_SIZE,
        color_to_class=COLOR_TO_CLASS,
    )

    print("\nPatches:")
    print(f"  Total: {len(image_patches)}")
    print(f"  Shape imagens: {image_patches.shape}")
    print(f"  Shape máscaras: {mask_patches.shape}")
    print(f"  Shape posições: {positions_norm.shape}")

    print("\nPrimeiro patch:")
    print(f"  Posição pixel: {positions_pixel[0]}")
    print(f"  Posição normalizada: {positions_norm[0]}")
    print(f"  Classe: {patch_labels[0]}")

    print("\nDistribuição das classes:")

    class_counts = Counter(patch_labels)

    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    labels_encoded = encode_labels(patch_labels)

    datasets = split_dataset(
        image_patches=image_patches,
        positions_norm=positions_norm,
        positions_pixel=positions_pixel,
        labels=labels_encoded,
    )

    print("\nDivisão dos dados:")
    print(f"  Total: {len(image_patches)}")
    print(f"  Treino: {len(datasets['X_img_train'])}")
    print(f"  Validação: {len(datasets['X_img_val'])}")
    print(f"  Teste: {len(datasets['X_img_test'])}")

    print("\nShapes:")
    print(f"  X treino: {datasets['X_img_train'].shape}")
    print(f"  Posições treino: {datasets['X_pos_train'].shape}")
    print(f"  y treino: {datasets['y_train'].shape}")

    print("\nDistribuição do treino ANTES do augmentation:")

    unique, counts = np.unique(
        datasets["y_train"],
        return_counts=True,
    )

    for class_index, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[class_index]}: {count}")

    (
        X_img_train_aug,
        X_pos_train_aug,
        y_train_aug,
    ) = augment_training_data(
        images=datasets["X_img_train"],
        positions=datasets["X_pos_train"],
        labels=datasets["y_train"],
        target_per_class=AUGMENTATION_TARGET_PER_CLASS,
        random_seed=RANDOM_SEED,
    )

    print("\nDistribuição do treino DEPOIS do augmentation:")

    unique, counts = np.unique(
        y_train_aug,
        return_counts=True,
    )

    for class_index, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[class_index]}: {count}")

    print("\nShapes após augmentation:")
    print(f"  Imagens: {X_img_train_aug.shape}")
    print(f"  Posições: {X_pos_train_aug.shape}")
    print(f"  Labels: {y_train_aug.shape}")

    # Teste visual do Data Augmentation
    num_examples = 5

    original_examples = datasets["X_img_train"][:num_examples]

    augmented_examples = np.asarray(
        [
            transform_basic(image=image)["image"]
            for image in original_examples
        ]
    )

    # Se quiser ver os plots só descomentar as linhas abaixo
    # plot_augmentation_examples(
    #     original_images=original_examples,
    #     augmented_images=augmented_examples,
    #     num_examples=num_examples,
    # )

    print("\nCriando modelo...")

    model = create_model(
        patch_size=PATCH_SIZE,
        num_classes=len(CLASS_NAMES),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        num_layers=NUM_TRANSFORMER_LAYERS,
        dropout=TRANSFORMER_DROPOUT,
    )

    model.summary()

    model_path = MODELS_DIR / "best_model.keras"

    if TRAIN_MODEL:
        print("\nIniciando treinamento...")

        history = train_model(
            model=model,
            X_img_train=X_img_train_aug,
            X_pos_train=X_pos_train_aug,
            y_train=y_train_aug,
            X_img_val=datasets["X_img_val"],
            X_pos_val=datasets["X_pos_val"],
            y_val=datasets["y_val"],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            model_path=model_path,
            early_stopping_monitor=EARLY_STOPPING_MONITOR,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )

        print("\nTreinamento concluído.")

        print(f"Épocas executadas: {len(history.history['loss'])}")

        print(f"Loss final: {history.history['loss'][-1]:.4f}")
        print(f"Accuracy final: {history.history['accuracy'][-1]:.4f}")

        print(f"Validation loss final: {history.history['val_loss'][-1]:.4f}")
        print(f"Validation accuracy final: {history.history['val_accuracy'][-1]:.4f}")

        plot_training_history(
            history,
            OUTPUTS_DIR / "figures",
        )

        print(f"\nMelhor modelo salvo em: {model_path}")

    else:
        print("\nCarregando modelo treinado...")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}. "
                "Execute o treinamento primeiro."
            )

        model = load_model(model_path)

        print("Modelo carregado com sucesso.")


    # -------------------------------------------------
    # AVALIAÇÃO NO CONJUNTO DE TESTE
    # -------------------------------------------------

    print("\nAvaliando modelo no conjunto de teste...")

    results = evaluate_model(
        model=model,
        X_img_test=datasets["X_img_test"],
        X_pos_test=datasets["X_pos_test"],
        y_test=datasets["y_test"],
        class_names=CLASS_NAMES,
    )

    plot_confusion_matrix(
        conf_matrix=results["confusion_matrix"],
        class_names=CLASS_NAMES,
        output_dir=OUTPUTS_DIR / "figures",
    )

    print("\nMatriz de confusão salva com sucesso.")

    # -------------------------------------------------
    # RECONSTRUÇÃO DA SEÇÃO SÍSMICA
    # -------------------------------------------------

    print("\nIniciando reconstrução da seção sísmica...")

    # Mapeamento classe -> cor
    class_to_color = {
        class_name: color
        for color, class_name in COLOR_TO_CLASS.items()
    }

    # Divisão da seção inteira, incluindo as bordas
    reconstruction_patches, reconstruction_positions_pixel, reconstruction_positions_norm = (
        divide_image_into_patches_with_positions(
            image=seismic_data,
            patch_size=PATCH_SIZE,
        )
    )

    print(f"Patches para reconstrução: {len(reconstruction_patches)}")

    # Classificação de todos os patches
    colored_patches = classify_and_color_patches(
        patches=reconstruction_patches,
        positions_norm=reconstruction_positions_norm,
        model=model,
        class_to_color=class_to_color,
        class_names=CLASS_NAMES,
        batch_size=RECONSTRUCTION_BATCH_SIZE,
    )

    # Reconstrução da imagem
    segmented_image = reconstruct_image_from_patches(
        patches=colored_patches,
        positions_pixel=reconstruction_positions_pixel,
        image_shape=seismic_data.shape,
    )

    print(f"Shape da imagem reconstruída: {segmented_image.shape}")

    # Salvar reconstrução e máscara de referência
    plot_reconstruction(
        segmented_image=segmented_image,
        reference_mask=mask,
        output_dir=OUTPUTS_DIR / "figures",
    )

    print("\nReconstrução concluída.")

    # -------------------------------------------------
    # MAPAS SOFTMAX DA SEÇÃO ORIGINAL
    # -------------------------------------------------

    print("\nGerando probabilidades Softmax...")

    patch_softmax_outputs = get_patch_softmax_outputs_with_position(
        patches=reconstruction_patches,
        positions_norm=reconstruction_positions_norm,
        model=model,
        batch_size=RECONSTRUCTION_BATCH_SIZE,
    )

    softmax_images = reconstruct_softmax_images(
        patch_softmax_outputs=patch_softmax_outputs,
        positions_pixel=reconstruction_positions_pixel,
        image_shape=seismic_data.shape,
        patch_size=PATCH_SIZE,
    )

    confidence_image = reconstruct_confidence_image(
        patch_softmax_outputs=patch_softmax_outputs,
        positions_pixel=reconstruction_positions_pixel,
        image_shape=seismic_data.shape,
        patch_size=PATCH_SIZE,
    )

    plot_softmax_images(
        softmax_images=softmax_images,
        confidence_image=confidence_image,
        class_names=CLASS_NAMES,
        output_dir=OUTPUTS_DIR / "figures" / "original",
    )

    print("\nMapas Softmax salvos.")

    plot_softmax_combined(
        softmax_images=softmax_images,
        class_names=CLASS_NAMES,
        output_dir=OUTPUTS_DIR / "figures" / "original",
    )   

    # -------------------------------------------------
    # INFERÊNCIA EM UMA SEÇÃO SÍSMICA NÃO VISTA
    # -------------------------------------------------

    print("\nIniciando inferência na segunda linha sísmica...")

    predict_seismic_line(
        model=model,
        image_path=UNSEEN_IMAGE_PATH,
        output_dir=OUTPUTS_DIR / "figures" / "unseen",
    )

if __name__ == "__main__":
    main()