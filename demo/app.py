import gradio as gr
import torch
import requests
from PIL import Image
from timm.data import create_transform

# Prepare the model.
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import models


model = models.GHCS(pretrained=True) # can change different model name
model.eval()

# Prepare the transform.
transform = create_transform(input_size=224, crop_pct=model.default_cfg['crop_pct'])

# Download human-readable labels for ImageNet.
response = requests.get("https://git.pk/JkYN")
labels = response.text.split("\n")

## Prepare images for demo
def download_image(url, filename):
  response = requests.get(url)
  if response.status_code == 200:
    with open(filename, 'wb') as f:
      f.write(response.content)
    print("Image downloaded successfully.")
  else:
    print("Failed to download image.")

image_list = ["*.jpg", "Output*.jpg"]

for img in image_list:
  image_url = f"https://raw.githubusercont/mis/content/fo1/{img}"
  file_path = f"{img}"
  download_image(image_url, file_path)


def predict(inp):
  inp = transform(inp).unsqueeze(0)

  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
  return confidences


title="GHCS:?"


gr.Interface(title=title,
             description=description,
             fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3),
             examples=image_list).launch()