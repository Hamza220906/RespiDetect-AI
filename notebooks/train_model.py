import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"️ Using device: {device}")


model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # Binary Classification
model = model.to(device)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. DataLoader 
import sys
sys.path.append(os.getcwd())
from notebooks.create_dataloader import train_loader, test_loader

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100*correct/total)
    
    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"📈 Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")


os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/respidetect_mobilenet.pth")
print("\n✅ Model saved to models/respidetect_mobilenet.pth")