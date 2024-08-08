import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import numpy as np

# 모델 로드 및 설정
model = models.efficientnet_b0(pretrained=True)
model.eval()

# 이미지 처리
def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 객체 감지 및 위치 추정
def detect_object(image_tensor):
    outputs = model(image_tensor)
    # 여기서는 단순화를 위해 최대 확률 위치를 사용합니다
    _, predicted = outputs.max(1)
    return predicted.item()

# 이미지 로드 및 처리
image_tensor = process_image('path_to_your_image.jpg')
object_presence = detect_object(image_tensor)

if object_presence:
    print("Object detected!")
    # 객체 위치 추정 로직 추가
else:
    print("No object detected.")
