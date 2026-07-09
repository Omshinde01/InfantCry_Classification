"""
evaluate.py
===========

Evaluate the trained Infant Cry Classification model.

Outputs:
--------
results/
    metrics.json
    classification_report.txt
    confusion_matrix.png
    predictions.csv
"""

import os
import json

import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from torch.utils.data import DataLoader

from dataset import BabyCryDataset
from model import InfantCryResNet
from config import *


def load_model():
    """Load trained model."""

    model = InfantCryResNet()

    model.load_state_dict(
        torch.load(
            BEST_MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


def evaluate_model(model, loader):
    """Run inference on dataset."""

    y_true = []
    y_pred = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())

    return y_true, y_pred


def save_metrics(y_true, y_pred):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted"
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted"
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted"
        ),
    }

    with open(
        os.path.join(RESULTS_DIR, "metrics.json"),
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    return metrics


def save_classification_report(y_true, y_pred):

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )

    with open(
        os.path.join(
            RESULTS_DIR,
            "classification_report.txt"
        ),
        "w"
    ) as file:

        file.write(report)


def save_predictions(y_true, y_pred):

    df = pd.DataFrame({
        "True Label": y_true,
        "Predicted Label": y_pred
    })

    df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "predictions.csv"
        ),
        index=False
    )


def save_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading dataset...")

    dataset = BabyCryDataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print("Loading model...")

    model = load_model()

    print("Running evaluation...")

    y_true, y_pred = evaluate_model(
        model,
        loader
    )

    metrics = save_metrics(
        y_true,
        y_pred
    )

    save_classification_report(
        y_true,
        y_pred
    )

    save_predictions(
        y_true,
        y_pred
    )

    save_confusion_matrix(
        y_true,
        y_pred
    )

    print("\nEvaluation Complete!\n")

    print("=" * 40)

    for key, value in metrics.items():

        print(f"{key:<12}: {value:.4f}")

    print("=" * 40)

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()