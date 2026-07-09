"""
app.py
======

Gradio interface for Infant Cry Classification.
"""

import gradio as gr

from predict import predict_audio


def classify(audio_file):
    """
    Classify an uploaded infant cry audio file.

    Args:
        audio_file (str): Path to uploaded audio.

    Returns:
        dict: Class probabilities (Gradio Label format)
    """

    if audio_file is None:
        return {"No Audio": 1.0}

    label, confidence = predict_audio(audio_file)

    return {
        label: confidence / 100,
        "Confidence": confidence / 100
    }


demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(
        sources=["upload"],
        type="filepath",
        label="Upload Infant Cry Audio"
    ),
    outputs=gr.Label(
        num_top_classes=5,
        label="Prediction"
    ),
    title="👶 Infant Cry Classification",
    description="""
Upload an infant cry recording to predict the reason behind the cry.

Classes:
• Hungry
• Tired
• Discomfort
• Belly Pain
• Burping
""",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()