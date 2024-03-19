import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report

input_size = 784
hidden_size1 = 128
hidden_size2 = 256  # 更好拟合，容易过拟合，计算困难
num_classes = 10
num_epochs = 10  # 增加可优化性能，但容易过拟合
batch_size = 500
learning_rate = 0.001

# 改数据格式
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_root = 'C:/Users/lenovo/Desktop/mnist'  # 替换mnist
train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform)
test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform)

# 设置验证集
validation_size = 10000  # 比如，我们从训练集中取出10000个样本作为验证集
train_size = len(train_dataset) - validation_size

# 分割数据集
train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 神经网络
class Net(nn.Module):
    def __init__(self, size, size1, size2, classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(size, size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(size1, size2)
        self.fc3 = nn.Linear(size2, classes)

    def forward(self, x):
        x = x.view(-1, input_size)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


model = Net(input_size, hidden_size1, hidden_size2, num_classes)
model = model.to(device)  # CPU换GPU


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.999)


# 开始训练
def train_model(model1, criterion1, optimizer1, scheduler1, num_epochs1, train_loader1, validation_loader1):
    train_losses = []
    val_losses = []
    val_accuracies = []
    model1.train()
    for epoch in range(num_epochs1):
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer1.zero_grad()
            outputs = model1(images)
            loss = criterion1(outputs, labels)
            loss.backward()
            optimizer1.step()
            scheduler1.step()
            train_loss += loss.item()
        train_loss /= len(train_loader1)
        train_losses.append(train_loss)

        model1.eval()  # 评估模式
        validation_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validation_loader1:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model1(images)
                loss = criterion1(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        validation_loss /= len(validation_loader1)
        accuracy = 100 * correct / total
        val_losses.append(validation_loss)
        val_accuracies.append(accuracy)
        print(
            f'Epoch [{epoch + 1}/{num_epochs1}], Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Accuracy: {accuracy:.2f}%')

        model1.train()  #训练模式
    return train_losses, val_losses, val_accuracies


# 进行评估
def evaluate_model(model1, test_loader1):
    model1.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader1:
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(-1, input_size)
            outputs = model1(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 计算四个值
    accuracy, precision, recall, f1_score = calculate_metrics(y_true, y_pred)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')
    print(classification_report(y_true, y_pred, digits=4))


def calculate_metrics(y_true, y_pred):
    y_true = torch.tensor(y_true, dtype=torch.int)
    y_pred = torch.tensor(y_pred, dtype=torch.int)
    TP = ((y_pred == 1) & (y_true == 1)).sum().item()
    FP = ((y_pred == 1) & (y_true == 0)).sum().item()
    TN = ((y_pred == 0) & (y_true == 0)).sum().item()
    FN = ((y_pred == 0) & (y_true == 1)).sum().item()
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return accuracy, precision, recall, f1_score


if __name__ == "__main__":
    train_losses, val_losses, val_accuracies = train_model(model, criterion, optimizer, scheduler, num_epochs,
                                                           train_loader, validation_loader)

    # 绘制训练损失和验证损失
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制验证准确率
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

