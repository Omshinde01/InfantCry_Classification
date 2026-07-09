"""
predict.py
==========

Inference module for Infant Cry Classification.
"""

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from model import InfantCryResNet
from config import (
    DEVICE,
    BEST_MODEL_PATH,
    TARGET_SAMPLE_RATE,
    AUDIO_DURATION,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    CLASS_LABELS,
)

# --------------------------------------------------------
# Load model once
# --------------------------------------------------------

model = InfantCryResNet()

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

# --------------------------------------------------------
# Audio transforms
# --------------------------------------------------------

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

db_transform = T.AmplitudeToDB()


def preprocess_audio(audio_path: str) -> torch.Tensor:
    """
    Convert an audio file into a normalized Mel Spectrogram.
    """

    waveform, sample_rate = sf.read(audio_path)

    waveform = torch.tensor(waveform, dtype=torch.float32)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T

    # Resample
    if sample_rate != TARGET_SAMPLE_RATE:

        resampler = T.Resample(
            sample_rate,
            TARGET_SAMPLE_RATE
        )

        waveform = resampler(waveform)

    # Stereo → Mono
    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    # Fix duration
    max_length = TARGET_SAMPLE_RATE * AUDIO_DURATION

    if waveform.shape[1] > max_length:

        waveform = waveform[:, :max_length]

    elif waveform.shape[1] < max_length:

        padding = max_length - waveform.shape[1]

        waveform = torch.nn.functional.pad(
            waveform,
            (0, padding)
        )

    mel = mel_transform(waveform)

    mel = db_transform(mel)

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    return mel.unsqueeze(0)


def predict_audio(audio_path: str):
    """
    Predict infant cry category.

    Returns
    -------
    tuple
        (predicted_class, confidence)
    """

    spectrogram = preprocess_audio(audio_path)

    spectrogram = spectrogram.to(DEVICE)

    with torch.no_grad():

        output = model(spectrogram)

        probabilities = F.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    prediction = prediction.item()

    confidence = confidence.item() * 100

    return (
        CLASS_LABELS[prediction],
        confidence
    )


if __name__ == "__main__":

    path = input("Audio Path : ")

    label, confidence = predict_audio(path)

    print()

    print(f"Prediction : {label}")

    print(f"Confidence : {confidence:.2f}%")