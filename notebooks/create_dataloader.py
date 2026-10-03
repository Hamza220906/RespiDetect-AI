import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


img_dir = "outputs/spectrograms"
audio_dir = "data/audio_files"

image_paths = [f for f in os.listdir(img_dir) if f.endswith(".png")]
labels = []

for img_name in image_paths:
    base_name = img_name.replace(".png", ".wav")
    txt_path = os.path.join(audio_dir, base_name.replace(".wav", ".txt"))
    
    is_diseased = 0
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4 and parts[3].strip() not in ["0", ""]:
                    is_diseased = 1
                    break
    labels.append(is_diseased)

# 2. Train/Test Split (80% Train, 20% Test)
train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"📊 Training Samples: {len(train_imgs)}")
print(f"📊 Testing Samples: {len(test_imgs)}")


class RespiratoryDataset(Dataset):
    def __init__(self, img_names, label_list, transform=None):
        self.img_names = img_names
        self.labels = label_list
        self.transform = transform
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


transform = transforms.Compose([
    transforms.Resize((224, 224)),      
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = RespiratoryDataset(train_imgs, train_labels, transform=transform)
test_dataset = RespiratoryDataset(test_imgs, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"\n✅ DataLoader Ready!")
print(f"   Batch Size: 32")
print(f"   Total Batches (Train): {len(train_loader)}")
print(f"   Total Batches (Test): {len(test_loader)}")


images, labels_batch = next(iter(train_loader))
print(f"\n🔍 Sample Batch Shape: {images.shape} | Labels: {labels_batch[:5]}")