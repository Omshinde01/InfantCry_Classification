"""
dataset.py
==========

PyTorch Dataset for Infant Cry Classification.

Converts raw audio into normalized Mel Spectrograms.
"""

import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import soundfile as sf

from torch.utils.data import Dataset

from config import (
    DATA_DIR,
    METADATA_PATH,
    TARGET_SAMPLE_RATE,
    AUDIO_DURATION,
    N_FFT,
    HOP_LENGTH,
    N_MELS
)


class BabyCryDataset(Dataset):
    """
    Infant Cry Dataset.

    Reads metadata.csv, loads audio,
    converts it to Mel Spectrogram,
    and returns (spectrogram, label).
    """

    def __init__(
        self,
        metadata_path: str = METADATA_PATH,
        data_dir: str = DATA_DIR,
    ):

        self.df = pd.read_csv(metadata_path)

        self.data_dir = data_dir

        self.target_sr = TARGET_SAMPLE_RATE

        self.max_length = TARGET_SAMPLE_RATE * AUDIO_DURATION

        self.resamplers = {}

        self.mel_transform = T.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

        self.db_transform = T.AmplitudeToDB()

    def __len__(self):

        return len(self.df)

    def _resample(self, waveform, sample_rate):

        if sample_rate == self.target_sr:
            return waveform

        if sample_rate not in self.resamplers:

            self.resamplers[sample_rate] = T.Resample(
                sample_rate,
                self.target_sr
            )

        return self.resamplers[sample_rate](waveform)

    def _convert_to_mono(self, waveform):

        if waveform.shape[0] > 1:

            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform

    def _fix_audio_length(self, waveform):

        if waveform.shape[1] > self.max_length:

            waveform = waveform[:, :self.max_length]

        elif waveform.shape[1] < self.max_length:

            padding = self.max_length - waveform.shape[1]

            waveform = nn.functional.pad(
                waveform,
                (0, padding)
            )

        return waveform

    def _create_mel_spectrogram(self, waveform):

        mel = self.mel_transform(waveform)

        mel = self.db_transform(mel)

        mel = (
            mel - mel.mean()
        ) / (
            mel.std() + 1e-6
        )

        return mel

    def __getitem__(self, index):

        row = self.df.iloc[index]

        audio_path = os.path.join(
            self.data_dir,
            row["filepath"]
        )

        label = int(row["label"])

        waveform, sample_rate = sf.read(audio_path)

        waveform = torch.tensor(waveform, dtype=torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T

        waveform = self._resample(
            waveform,
            sample_rate
        )

        waveform = self._convert_to_mono(
            waveform
        )

        waveform = self._fix_audio_length(
            waveform
        )

        mel = self._create_mel_spectrogram(
            waveform
        )

        return mel, label