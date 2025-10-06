
# 🚢 Ship Detector (Aerial Image Classifier)

A deep learning project that detects whether an **aerial image** contains a **ship** or not.  
Built using **FastAI** for training and **Gradio** for deployment.

---

## 🧠 Project Overview
This project uses convolutional neural networks (CNNs) to classify aerial satellite images into two categories:

- **Ship** — if a ship or boat is present  
- **NoShip** — if there is no ship in the image  

The model was trained on a custom dataset of aerial imagery and deployed with an interactive Gradio interface.

---

## ⚙️ Tech Stack
- **Python 3.11+**
- **FastAI 2.8+**
- **PyTorch**
- **Gradio** (for deployment)
- **PIL / NumPy / Pandas**

---

## 🧩 Dataset Structure

The dataset should be organized as:

```

Images/
├── Ship/
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
└── NoShip/
├── imageA.png
├── imageB.jpg
└── ...

````

FastAI automatically assigns labels (`Ship`, `NoShip`) using folder names.

---

## 🧠 Model Training (FastAI)
```python
from fastai.vision.all import *

path = Path('Images')

Ships = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128),
    batch_tfms=aug_transforms(mult=2)
)

dls = Ships.dataloaders(path)

learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(3)

# Save the trained model
learn.export('model.pkl')
````

---

## 🖥️ Gradio App Code

```python
from fastai.vision.all import *
import gradio as gr

# Load trained model
learn = load_learner('model.pkl')

# Prediction function
def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return {pred: float(probs[pred_idx])}

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=2),
    title="🚢 Ship Detector (Aerial Image Classifier)",
    description="Upload an aerial image to check if a ship is present."
)

iface.launch()
```

---

## 📊 Model Performance

| Metric              | Value  |
| ------------------- | ------ |
| Train Accuracy      | ~98%   |
| Validation Accuracy | ~98%   |
| Error Rate          | ~0.016 |

✅ The model generalizes well with no overfitting observed.

---

## 🧪 How to Run Locally

1. Clone or download this project.
2. Install dependencies:

   ```bash
   pip install fastai gradio
   ```
3. Place your `model.pkl` file in the project directory.
4. Run:

   ```bash
   python app.py
   ```
5. A Gradio web app will open in your browser. 🎉

---

## 🌍 Future Improvements

* Add segmentation to detect ship **location**, not just presence.
* Train on larger satellite datasets (e.g., Airbus Ship Detection).
* Deploy on **Hugging Face Spaces** or **Streamlit Cloud**.

---

## ✨ Author

**Macha (Sandy)**
🧑‍💻 Data Scientist | Deep Learning Enthusiast
📫 *Feel free to connect for collaborations or learning discussions!*

```

