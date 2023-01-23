import torch
from torchvision import transforms
from fastapi import FastAPI, File, Form, HTTPException, Response, status
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = torch.jit.load("./model.pt")
model.eval()
model.to("cpu")

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def get_res(image):
    cls = ["cloudy", "rain", "shine", "sunrise"]
    image = _transform(image).unsqueeze(0).to("cpu")
    with torch.no_grad():
        res = model(image).cpu().numpy()
    return cls[np.argmax(res)]


img = Image.open(io.BytesIO('./test.jpg'))
res = get_res(img)
print(res)

