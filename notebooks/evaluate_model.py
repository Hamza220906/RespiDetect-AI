import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("models/respidetect_mobilenet.pth", map_location=device))
model = model.to(device)
model.eval()


import sys
sys.path.append(os.getcwd())
from notebooks.create_dataloader import test_loader


all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


print("\n📊 TEST SET EVALUATION RESULTS:")
print("="*50)
print(classification_report(
    all_labels, all_preds, 
    target_names=["Healthy (0)", "Diseased (1)"],
    digits=4
))

cm = confusion_matrix(all_labels, all_preds)
print(f"Confusion Matrix:\n{cm}")
print(f"\n✅ Test Accuracy: {np.mean(np.array(all_preds) == np.array(all_labels))*100:.2f}%")