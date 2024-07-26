import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from torch import nn, optim



if __name__ == "__main__":
    # 데이터셋 경로 설정
    train_directory = '/home/hdd/student/intern/배선 고정 불량/훈련데이터'
    val_directory = '/home/hdd/student/intern/배선 고정 불량/검증데이터'

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 로드
    train_dataset = torchvision.datasets.ImageFolder(root=train_directory, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=val_directory, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 모델 정의
    model = efficientnet_b0(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # 모델의 사전 학습된 파라미터를 고정

    # 분류기 교체 (EfficientNet의 출력 계층을 조정)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # 두 개의 클래스: 양품, 불량

    # CUDA 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 손실 함수 및 최적화기
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # 학습 함수
    def train_model(model, criterion, optimizer, num_epochs=10):
        model.train()
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # 모델 학습
    train_model(model, criterion, optimizer, num_epochs=10)

    # 모델 저장
    torch.save(model.state_dict(), 'efficientnet_b0.pth')

    print("모델 학습 및 저장 완료!")

