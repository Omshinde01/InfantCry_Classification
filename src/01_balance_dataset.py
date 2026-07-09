"""
01_balance_dataset.py
=====================

Scans the raw dataset, balances all classes using random
oversampling, and generates metadata.csv.

Dataset Structure
-----------------
data/
│
├── raw/
│   ├── hungry/
│   ├── tired/
│   ├── discomfort/
│   ├── belly_pain/
│   └── burping/
│
└── balanced/
"""

import os
import random

from config import (
    RAW_DATA_DIR,
    BALANCED_DATA_DIR,
    METADATA_PATH,
    CLASS_NAMES,
    CLASS_TO_LABEL,
    SUPPORTED_AUDIO_EXTENSIONS,
    REMOVE_EXISTING_BALANCED_DATA,
)

from utils import (
    create_directory,
    copy_audio,
    get_audio_files,
    print_dataset_statistics,
    print_header,
    print_info,
    print_success,
    print_error,
    save_metadata,
    set_random_seed,
)


def scan_dataset():
    """
    Scan dataset and return all audio files.

    Returns
    -------
    dict
        {
            "hungry": [...],
            ...
        }
    """

    dataset = {}

    for class_name in CLASS_NAMES:

        class_dir = os.path.join(RAW_DATA_DIR, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Missing folder : {class_dir}"
            )

        dataset[class_name] = get_audio_files(
            class_dir,
            SUPPORTED_AUDIO_EXTENSIONS
        )

    return dataset


def validate_dataset(dataset):
    """
    Ensure every class contains at least one file.
    """

    for class_name, files in dataset.items():

        if len(files) == 0:
            raise ValueError(
                f"No audio found inside '{class_name}'"
            )


def create_balanced_dataset(dataset):
    """
    Oversample every class to match the largest class.
    """

    target_size = max(len(files) for files in dataset.values())

    print_info(
        f"Target samples per class : {target_size}"
    )

    metadata = []

    total = 0

    for class_name, files in dataset.items():

        label = CLASS_TO_LABEL[class_name]

        output_dir = os.path.join(
            BALANCED_DATA_DIR,
            class_name
        )

        create_directory(output_dir)

        # -------------------------------------------------
        # Copy original files
        # -------------------------------------------------

        copied = 0

        for file_path in files:

            filename = os.path.basename(file_path)

            destination = os.path.join(
                output_dir,
                filename
            )

            copy_audio(file_path, destination)

            metadata.append({
                "filepath": destination,
                "filename": filename,
                "class_name": class_name,
                "label": label,
                "duplicated": False
            })

            copied += 1
            total += 1

        # -------------------------------------------------
        # Oversample remaining files
        # -------------------------------------------------

        while copied < target_size:

            source = random.choice(files)

            original_name = os.path.basename(source)

            name, ext = os.path.splitext(original_name)

            duplicate_name = (
                f"{name}_copy{copied-len(files)+1}{ext}"
            )

            destination = os.path.join(
                output_dir,
                duplicate_name
            )

            copy_audio(source, destination)

            metadata.append({
                "filepath": destination,
                "filename": duplicate_name,
                "class_name": class_name,
                "label": label,
                "duplicated": True
            })

            copied += 1
            total += 1

    save_metadata(metadata, METADATA_PATH)

    return total


def main():

    print_header("STEP 01 : BALANCE DATASET")

    set_random_seed()

    create_directory(
        BALANCED_DATA_DIR,
        REMOVE_EXISTING_BALANCED_DATA
    )

    print_info("Scanning dataset...")

    dataset = scan_dataset()

    validate_dataset(dataset)

    print_dataset_statistics(dataset)

    total = create_balanced_dataset(dataset)

    print_success(
        f"Balanced dataset created successfully."
    )

    print_success(
        f"Total files : {total}"
    )

    print_success(
        f"Metadata saved to : {METADATA_PATH}"
    )


if __name__ == "__main__":
    main()