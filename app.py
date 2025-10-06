from fastai.vision.all import *
import gradio as gr

# Load model
learn = load_learner('export.pkl')

# Prediction function
def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return {pred: float(probs[pred_idx])}

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=2),
    title="🚢 Ship Detector (Aerial Image Classifier)",
    description="Upload an aerial image, and the model will predict whether a ship is present or not."
)

# Launch
iface.launch()
