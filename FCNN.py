import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor，并归一化到[0, 1]
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 定义全连接神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 第一层：输入层到隐藏层
        self.fc2 = nn.Linear(128, 64)     # 第二层：隐藏层到隐藏层
        self.fc3 = nn.Linear(64, 10)      # 第三层：隐藏层到输出层
        self.relu = nn.ReLU()             # ReLU激活函数
        self.softmax = nn.Softmax(dim=1)  # Softmax激活函数，用于输出层

    def forward(self, x):
        x = x.view(-1, 28*28)  # 将每张图片展平为一维向量
        x = self.relu(self.fc1(x))  # 第一层到ReLU
        x = self.relu(self.fc2(x))  # 第二层到ReLU
        x = self.fc3(x)            # 第三层，输出层
        return self.softmax(x)     # Softmax输出

model = SimpleNN()

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 4. 训练模型
def train(model, criterion, optimizer, train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# 5. 评估模型
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# 执行训练和评估
train(model, criterion, optimizer, train_loader)
evaluate(model, test_loader)