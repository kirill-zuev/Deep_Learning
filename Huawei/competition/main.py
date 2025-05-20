import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm

data_train = './data/train_prepr/'
csv_file = './data/train.csv'

df = pd.read_csv(csv_file)

class CustomDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.data_dir, row['image_name'])
        image = Image.open(img_path).convert('RGB')
        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(df, data_train, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for b, l in train_loader:
    print(b.shape)
    print(l)
    break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = timm.create_model('convnext_xlarge', pretrained=True)
model.head.fc = nn.Linear(model.head.in_features, 46)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    best_acc = 0

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()

    train_acc = correct / len(train_loader.dataset)
    train_loss = running_loss / len(train_loader)
    
    if train_acc > best_acc:
        torch.save(model.state_dict(), 'convnext_model.pth')
        best_acc = train_acc
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')