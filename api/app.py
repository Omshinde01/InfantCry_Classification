import io
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import boto3

import tensorflow as tf
import tensorflow_hub as hub

import torch
import torchaudio.transforms as T
import torch.nn as nn
import torchvision.models as models

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse

app = FastAPI(title="Infant Cry & Noise Diagnostic API")

# =====================================================
# AWS S3 STORAGE SETUP
# =====================================================
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "your-baby-cry-history-bucket")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def upload_to_s3_background(file_bytes: bytes, unique_filename: str):
    """Asynchronously archives all incoming audio packets to S3 to eliminate latency."""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"recordings/{unique_filename}",
            Body=file_bytes,
            ContentType="audio/wav"
        )
        print(f"[S3 Success] Archived {unique_filename} to cloud history.")
    except Exception as e:
        print(f"[S3 Error] Background tracking upload failed: {str(e)}")


# =====================================================
# CORE AI MODEL MACHINE INITIALIZATION
# =====================================================
MODEL_PATH = "best_infant_cry_resnet.pth"
COMBINED_CRY_THRESHOLD = 0.10
TARGET_SR = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize Stage 1 (YAMNet)
print("Loading YAMNet Gatekeeper...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
class_names = pd.read_csv(class_map_path)["display_name"].tolist()

KEYWORDS = ["baby", "infant", "cry", "crying", "sob", "whimper", "wail", "moan", "scream", "child"]
cry_classes = [(idx, name) for idx, name in enumerate(class_names) if any(k in name.lower() for k in KEYWORDS)]

# 2. ResNet34 Structural Definition
class InfantCryResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        orig_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, orig_conv.out_channels, kernel_size=orig_conv.kernel_size, stride=orig_conv.stride, padding=orig_conv.padding, bias=False)
        self.backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.backbone.fc.in_features, num_classes))
    def forward(self, x): return self.backbone(x)

# 3. Initialize Stage 2 (Custom ResNet)
print("Loading ResNet Classifier...")
model = InfantCryResNet(num_classes=5)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print(f"WARNING: {MODEL_PATH} not found. Running structural verify mode.")
model.to(device).eval()

# Pre-allocate Mel transformations to save processing overhead
mel_transform = T.MelSpectrogram(sample_rate=TARGET_SR, n_fft=1024, hop_length=512, n_mels=64)
class_names_model = ["Hungry", "Tired", "Discomfort", "Belly Pain", "Burping"]

print("All models loaded. System is operational.")


# =====================================================
# HIGH-SPEED API PREDICTION PIPELINE
# =====================================================

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Read files into RAM buffer directly
        file_bytes = await file.read()
        
        # Fire off asynchronous cloud storage tracking process
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}.wav"
        background_tasks.add_task(upload_to_s3_background, file_bytes, unique_filename)

        # Decode audio arrays in memory
        with io.BytesIO(file_bytes) as fp:
            waveform, sr = sf.read(fp)
            
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

        # -------------------------------------------------
        # STAGE 1: YAMNet Auditory Evaluation
        # -------------------------------------------------
        scores, _, _ = yamnet(waveform.astype(np.float32))
        mean_scores = scores.numpy().mean(axis=0)
        combined_score = sum(mean_scores[idx] for idx, _ in cry_classes)

        # IF NO BABY CRY IS VERIFIED: Fetch top 3 ambient noises instead
        if combined_score <= COMBINED_CRY_THRESHOLD:
            top3_indices = mean_scores.argsort()[-3:][::-1]
            top_noises = {class_names[idx]: round(float(mean_scores[idx]) * 100, 2) for idx in top3_indices}
            
            return {
                "cry_detected": 0,
                "combined_cry_score": float(combined_score),
                "prediction": "Noise Detected",
                "archive_name": unique_filename,
                "top_noises": top_noises
            }

        # -------------------------------------------------
        # STAGE 2: Deep-Analysis Diagnostic ResNet
        # -------------------------------------------------
        max_len = TARGET_SR * 5
        if len(waveform) > max_len:
            waveform = waveform[:max_len]
        elif len(waveform) < max_len:
            waveform = np.pad(waveform, (0, max_len - len(waveform)))

        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        mel_spec = mel_transform(waveform_tensor)
        mel_spec = T.AmplitudeToDB()(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        input_tensor = mel_spec.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        breakdown = {class_names_model[i]: round(probs[i].item() * 100, 2) for i in range(len(class_names_model))}
        
        return {
            "cry_detected": 1,
            "combined_cry_score": float(combined_score),
            "prediction": class_names_model[pred_idx],
            "archive_name": unique_filename,
            "breakdown": breakdown
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)