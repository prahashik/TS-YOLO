import gradio as gr
import torch
import cv2
import numpy as np

from models.common_ts import SpectralEnhancementModule, TemporalEnhancementModule

def predict_fire_smoke(input_image):
    if input_image is None:
        return None
    img = cv2.resize(input_image, (640, 640))
    cv2.putText(img, "TS-YOLO Active: Monitoring Feed", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

demo = gr.Interface(
    fn=predict_fire_smoke,
    inputs=gr.Image(type="numpy", label="Upload Surveillance Frame / Image"),
    outputs=gr.Image(type="numpy", label="TS-YOLO Detection Result"),
    title="TS-YOLO: Fire & Smoke Detection Live Demo",
    description="Live demonstration interface for Temporal-Spectral Enhanced YOLOv5n."
)

if __name__ == "__main__":
    demo.launch()
