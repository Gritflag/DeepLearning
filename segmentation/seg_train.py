import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import json
from PIL import Image
import numpy as np

class CocoSegmentationDataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation) as f:
            self.annotations = json.load(f)
        self.images = self.annotations['images']
        self.annotations = self.annotations['annotations']

    def __getitem__(self, idx):
        img_id = self.images[idx]['id']
        img_path = f"{self.root}/{self.images[idx]['file_name']}"
        img = Image.open(img_path).convert("RGB")
        
        # COCO dataset uses polygons to represent segmentation masks
        mask = Image.new('L', (img.width, img.height))
        for annotation in [anno for anno in self.annotations if anno['image_id'] == img_id]:
            ImageDraw.Draw(mask).polygon(annotation['segmentation'][0], outline=1, fill=1)
        mask = np.array(mask)

        if self.transforms:
            img = self.transforms(img)
            mask = torch.as_tensor(mask, dtype=torch.uint8)

        return img, mask

    def __len__(self):
        return len(self.images)

class EfficientNetUNet(nn.Module):
    def __init__(self):
        super(EfficientNetUNet, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True).features
        
        self.upconv1 = nn.ConvTranspose2d(1280, 320, 2, stride=2)
        self.conv1 = nn.Conv2d(320, 320, 3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(320, 112, 2, stride=2)
        self.conv2 = nn.Conv2d(112, 112, 3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(112, 40, 2, stride=2)
        self.conv3 = nn.Conv2d(40, 40, 3, padding=1)
        
        self.upconv4 = nn.ConvTranspose2d(40, 24, 2, stride=2)
        self.conv4 = nn.Conv2d(24, 24, 3, padding=1)

        self.final_conv = nn.Conv2d(24, 1, 1)
    
    def forward(self, x):
        x = self.efficientnet(x)
        
        x = self.upconv1(x)
        x = self.conv1(x)
        
        x = self.upconv2(x)
        x = self.conv2(x)
        
        x = self.upconv3(x)
        x = self.conv3(x)
        
        x = self.upconv4(x)
        x = self.conv4(x)

        x = self.final_conv(x)
        return x

# 데이터셋 및 DataLoader 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CocoSegmentationDataset('path_to_images', 'path_to_annotations.json', transforms=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 모델 및 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetUNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
for epoch in range(10):  # 10 에포크로 설정
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, masks.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
