import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from metodologia_proposta.train import prepare_images


def evaluate_model(
    model,
    X_img_test,
    X_pos_test,
    y_test,
    class_names,
):
    """
    Avalia o modelo no conjunto de teste e retorna
    a matriz de confusão.
    """

    X_img_test = prepare_images(X_img_test)

    # Avaliação
    test_loss, test_acc = model.evaluate(
        [X_img_test, X_pos_test],
        y_test,
        verbose=1,
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Predições
    probabilities = model.predict(
        [X_img_test, X_pos_test],
        verbose=1,
    )

    y_pred = np.argmax(probabilities, axis=1)

    # Relatório
    print("\nRelatório de classificação:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )

    # Matriz de confusão
    conf_matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    return {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "y_pred": y_pred,
        "confusion_matrix": conf_matrix,
    }