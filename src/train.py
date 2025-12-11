import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# --- 1. Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 2. Load dataset 
train_data = datasets.ImageFolder("dataset/", transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

# --- 3. Load pretrained model ---
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(1280, 6)  # 6 classes (1,5,10,20,50,100)

model = model.to("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#  Training loop 
for epoch in range(5):
    for images, labels in train_loader:
        images, labels = images.to("cuda"), labels.to("cuda")

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss={loss.item()}")

# --- 5. Save model ---
torch.save(model.state_dict(), "models/currency_model.pth")
print("Model saved!")
