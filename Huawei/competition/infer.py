import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import timm
from tqdm import tqdm

data_test = './data/test_prepr/'

image_paths = []
for root, dirs, files in os.walk(data_test):
    for file in files:
        if file.lower().endswith('.jpg'):
            image_paths.append(os.path.join(root, file))

image_paths = sorted(image_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
df = pd.DataFrame(image_paths, columns=['image_name'])
print(df)

class CustomDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['image_name']
        image = Image.open(img_path).convert('RGB')
        print(img_path)

        if self.transform:
            image = self.transform(image)

        return image

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = CustomDataset(df, data_test, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = timm.create_model('convnext_xlarge', pretrained=False)
model.head.fc = nn.Linear(model.head.fc.in_features, 46)
model.load_state_dict(torch.load('./convnext_model.pth'))
model = model.to(device)
model.eval()

output_file = 'submission.csv'
results = []
with torch.no_grad():
    i = 1
    for images in tqdm(test_loader, desc="Testing"):
        images = images.to(device)

        outputs = model(images)
        preds = outputs.argmax(1).item()

        results.append([i, preds])
        i += 1
df = pd.DataFrame(results, columns=["id", "label"])
df.to_csv(output_file, index=False)