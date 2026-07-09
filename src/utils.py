"""
Utility Functions
=================

Common utility functions used throughout the Infant Cry
Classification pipeline.
"""

import os
import shutil
import random
import pandas as pd
import numpy as np
import torch

from config import RANDOM_SEED


# ==========================================================
# DIRECTORY UTILITIES
# ==========================================================

def create_directory(directory: str, remove_existing: bool = False) -> None:
    """
    Create a directory.

    Args:
        directory: Directory path.
        remove_existing: Remove existing directory before creating.
    """

    if remove_existing and os.path.exists(directory):
        shutil.rmtree(directory)

    os.makedirs(directory, exist_ok=True)


# ==========================================================
# RANDOMNESS
# ==========================================================

def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)


# ==========================================================
# DATASET UTILITIES
# ==========================================================

def get_audio_files(directory: str, extensions: tuple) -> list:
    """
    Return all audio files inside a directory.

    Args:
        directory: Directory path.
        extensions: Supported extensions.

    Returns:
        List of audio file paths.
    """

    audio_files = []

    for file in os.listdir(directory):

        path = os.path.join(directory, file)

        if os.path.isfile(path) and file.lower().endswith(extensions):
            audio_files.append(path)

    return sorted(audio_files)


def count_audio_files(directory: str, extensions: tuple) -> int:
    """
    Count number of audio files inside a directory.
    """

    return len(get_audio_files(directory, extensions))


# ==========================================================
# METADATA
# ==========================================================

def save_metadata(records: list, csv_path: str) -> None:
    """
    Save metadata as CSV.

    Args:
        records: List of dictionaries.
        csv_path: Output csv path.
    """

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

def save_model(model, path):
    """
    Save model weights.
    """
    torch.save(model.state_dict(), path)
# ==========================================================
# PRINT UTILITIES
# ==========================================================

def print_header(title: str) -> None:
    """
    Print section header.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_success(message: str) -> None:
    print(f"[SUCCESS] {message}")


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def print_warning(message: str) -> None:
    print(f"[WARNING] {message}")


def print_error(message: str) -> None:
    print(f"[ERROR] {message}")


# ==========================================================
# DATASET STATISTICS
# ==========================================================

def print_dataset_statistics(dataset: dict) -> None:
    """
    Print number of files per class.

    Args:
        dataset:
            {
                class_name: [list_of_audio_files]
            }
    """

    print("\nDataset Statistics")
    print("-" * 35)

    total = 0

    for class_name, files in dataset.items():

        count = len(files)

        total += count

        print(f"{class_name:<15} : {count}")

    print("-" * 35)
    print(f"{'Total':<15} : {total}")


# ==========================================================
# FILE COPYING
# ==========================================================

def copy_audio(source: str, destination: str) -> None:
    """
    Copy audio file preserving metadata.
    """

    shutil.copy2(source, destination)