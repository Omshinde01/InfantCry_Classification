"""
train.py
========

Training pipeline for Infant Cry Classification.
"""

import torch
import torch.nn as nn

from sklearn.metrics import f1_score

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import BabyCryDataset
from model import InfantCryResNet

from config import *

from utils import save_model

def create_dataloaders():

    dataset = BabyCryDataset()

    train_size = int(TRAIN_SPLIT * len(dataset))

    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(

        dataset,

        [train_size, val_size],

        generator=torch.Generator().manual_seed(
            RANDOM_SEED
        )
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        drop_last=True

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY

    )

    return train_loader, val_loader

def create_optimizer(model):

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=LR_FACTOR,

        patience=LR_PATIENCE

    )

    return optimizer, scheduler

criterion = nn.CrossEntropyLoss()

def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer
):

    model.train()

    running_loss = 0

    predictions = []

    targets = []

    for inputs, labels in loader:

        inputs = inputs.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        predictions.extend(

            outputs.argmax(1).cpu().numpy()

        )

        targets.extend(

            labels.cpu().numpy()

        )

    loss = running_loss / len(loader.dataset)

    f1 = f1_score(

        targets,

        predictions,

        average="weighted"

    )

    return loss, f1

def validate(
    model,
    loader,
    criterion
):
    """
    Validate the model for one epoch.

    Args:
        model: PyTorch model.
        loader: Validation DataLoader.
        criterion: Loss function.

    Returns:
        tuple:
            (validation_loss, validation_f1)
    """

    model.eval()

    running_loss = 0.0

    predictions = []

    targets = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())

            targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    epoch_f1 = f1_score(
        targets,
        predictions,
        average="weighted"
    )

    return epoch_loss, epoch_f1
    
def train():

    train_loader, val_loader = create_dataloaders()

    model = InfantCryResNet().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler = create_optimizer(model)

    best_f1 = 0.0

    print("\nStarting Training...\n")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_f1 = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train F1: {train_f1:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1

            save_model(
                model,
                BEST_MODEL_PATH
            )

            print(
                f"✅ Best model saved! "
                f"(Val F1 = {best_f1:.4f})"
            )

    print("\nTraining Complete!")
    print(f"Best Validation F1: {best_f1:.4f}")

if __name__ == "__main__":
    train()