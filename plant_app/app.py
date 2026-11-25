import json
import numpy as np
import torch, timm
import gradio as gr
from PIL import Image

classes = json.load(open("classes.json"))
model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load("student.pt", map_location="cpu"))
model.eval()

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pretty(c):
    plant, _, cond = c.partition("___")
    cond = cond.replace("_", " ").strip()
    return f"{plant.replace('_', ' ').replace(',', '').strip()} - {cond}"


def predict(img):
    if img is None:
        return {}
    x = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    v, i = p.topk(3)
    return {pretty(classes[k]): float(s) for s, k in zip(v, i)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Leaf photo"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Plant Disease Detector",
    description=(
        "Upload a photo of a single leaf. A small MobileNetV3 model distilled from "
        "BioCLIP-2 and DINOv2 predicts the plant and disease (38 PlantVillage classes). "
        "Works best on one leaf against a plain background; accuracy on cluttered field "
        "photos is much lower (about 31% in our tests). Not a substitute for expert diagnosis."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
