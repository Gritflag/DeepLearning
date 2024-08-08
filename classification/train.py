import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

def load_model(name, num_classes, pretrained):
    """EfficientNet-B0 모델을 로드하고 수정합니다."""
    if not pretrained:
        if name == "poorWiringFixation.pth":
            model = efficientnet_b0(pretrained=False)
            model.load_state_dict(torch.load(name, map_location=device))
        else:
            model = efficientnet_b0(pretrained=False)
    else:
        model = efficientnet_b0(pretrained=True)
        
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    model.eval()
    return model

def train_validate(model, trainloader, valloader, criterion, optimizer, writer, device, epochs=10):
    """모델을 훈련시키고 검증합니다."""
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(trainloader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(valloader)
        accuracy = 100 * correct / total
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}%")

if __name__ == "__main__":
    data_dir = '/home/hdd/student/intern/casting/data'
    # 데이터 변환을 위한 업데이트된 변환 코드
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지 크기 조정
        transforms.RandomHorizontalFlip(),  # 무작위 수평 뒤집기
        transforms.RandomRotation(10),  # 최대 10도 회전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 지터
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 정규화
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    torch.manual_seed(42)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")   ##GPU 사용 설정
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    model_names = ["efficientnet_b0_weights_pretrained.pth", "poorWiringFixation.pth", "efficientnet_b0_weights_notPretrained.pth"]
    pretrained_status = [True, False, False]
    titles = ["Pretrained Model", "Custom Trained Model", "Not Pretrained Model"]

    for i, model_name in enumerate(model_names):
        writer = SummaryWriter(f'runs/{titles[i]}')
        print(titles[i])
        model = load_model(model_name, len(dataset.classes), pretrained_status[i])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        train_validate(model, trainloader, valloader, criterion, optimizer, writer, device, epochs=100)
        writer.close()
