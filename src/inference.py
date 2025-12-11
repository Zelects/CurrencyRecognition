import torch
from torchvision import models, transforms
from PIL import Image

classes = ["1", "5", "10", "20", "50", "100"]

def predict(path):
    model = models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(1280, 6)
    model.load_state_dict(torch.load("models/currency_model.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(path)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = outputs.max(1)
        print("Predicted denomination:", classes[pred.item()])

predict("dataset/50/360_F_380276559_I0tXMtHxccgfXbLozWMO4PWSVKHd9YwH.jpg")
