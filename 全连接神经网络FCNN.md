### 什么是全连接神经网络？

全连接神经网络（FCNN）是最基础的神经网络结构，它由多个神经元组成，这些神经元按照层级顺序连接在一起。每一层的每个神经元都与前一层的每个神经元连接。

想象你在参加一个盛大的晚会，晚会上有三个区域：**接待区**、**交流区**和**结果区**。

- **接待区**（输入层）：负责接收来宾（数据），每个来宾代表一个特征。
- **交流区**（隐藏层）：每个来宾在交流区与其他来宾交流，交换信息。这里的每个交流区的来宾（神经元）都会与其他来宾进行对话，以获得更深层次的理解。
- **结果区**（输出层）：最后，在结果区，来宾们会得出他们的总结（预测结果），然后将其提供给晚会的组织者（输出）。

### 神经网络的结构

#### 输入层（Input Layer）

- **功能**：接受原始数据。每个神经元代表一个特征，例如图片的像素值、语音信号的特征等。
- **示例**：如果你有一张28x28像素的灰度图片，那么输入层会有784个神经元（28x28=784），每个神经元代表一个像素值。

#### 隐藏层（Hidden Layer）

- **功能**：处理和提取数据特征。每个神经元通过加权和激活函数来处理输入数据，然后将结果传递到下一层。
- **示例**：隐藏层的神经元数目可以是任意的，例如100个神经元。隐藏层能够提取数据的复杂特征，如图像中的边缘或形状。

#### 输出层（Output Layer）

- **功能**：给出最终的预测结果。输出层的神经元数目等于任务的类别数。例如，在数字分类任务中，输出层有10个神经元（分别代表0到9这10个数字）。
- **示例**：如果你要识别图片中的数字（0-9），输出层的每个神经元会输出一个数字的概率。

#### 激活函数（Activation Function）

- **功能**：引入非线性，使得神经网络能够处理复杂的模式。激活函数决定了神经元是否激活。
- **常见激活函数**：
  - **ReLU**（Rectified Linear Unit）：`f(x) = max(0, x)`。用于隐藏层，能够引入非线性，提升模型表现。
  - **Sigmoid**：`f(x) = 1 / (1 + exp(-x))`。用于输出层，特别是在二分类任务中。

### 下面为代码案例

我们使用MNIST数据集来训练一个简单的全连接神经网络模型。MNIST数据集包含了手写数字的图像，每个图像的大小为28x28像素，共10个类别（0-9）。

如果在下载 MNIST 数据集时遇到了问题。尝试手动下载数据集（**经尝试此方法似乎会被墙，说没有权限下载**）

### 手动下载数据集

可以手动下载 MNIST 数据集，并将其放到合适的目录下。

1. **下载数据集**：
   - [MNIST 数据集](http://yann.lecun.com/exdb/mnist/) 主页提供了所有必要的文件。可以直接从这里下载：
     - `train-images-idx3-ubyte.gz`
     - `train-labels-idx1-ubyte.gz`
     - `t10k-images-idx3-ubyte.gz`
     - `t10k-labels-idx1-ubyte.gz`

2. **解压文件**：
   - 下载后，可以使用工具解压这些 `.gz` 文件。例如，在终端中运行：
     ```bash
     gunzip train-images-idx3-ubyte.gz
     gunzip train-labels-idx1-ubyte.gz
     gunzip t10k-images-idx3-ubyte.gz
     gunzip t10k-labels-idx1-ubyte.gz
     ```

3. **移动文件**：
   - 将解压后的文件放到项目目录下的 `./data/MNIST/raw/` 目录中。

### 1. 数据预处理

首先，我们需要对 MNIST 数据集进行预处理，将图像转换为 Tensor，并进行归一化处理。

```python
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
```

**讲解**：
- 使用 `torchvision.transforms` 对 MNIST 数据集进行转换，将图像转换为 Tensor，并归一化到 [0, 1] 范围。
- 将数据集加载到 `DataLoader` 中，设置批量大小和是否打乱数据。

### 2. 定义全连接神经网络模型

接下来，定义一个简单的全连接神经网络模型。这个模型包括三个全连接层和两个激活函数（ReLU 和 Softmax）。

```python
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
```

**讲解**：
- 定义了一个名为 `SimpleNN` 的神经网络类，继承自 `nn.Module`。
- 在 `__init__` 方法中定义了三个全连接层（`fc1`, `fc2`, `fc3`）和两个激活函数（`ReLU` 和 `Softmax`）。
- 在 `forward` 方法中定义了数据如何流经网络层，包括展平、激活函数和最终的 Softmax 输出。

### 3. 定义损失函数和优化器

然后，我们定义损失函数和优化器。这里使用交叉熵损失函数和 Adam 优化器。

```python
# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
```

**讲解**：
- 使用 `nn.CrossEntropyLoss` 作为损失函数，这适用于分类任务。
- 使用 `optim.Adam` 作为优化器来更新网络参数，设置学习率为 0.001。

### 4. 训练模型

接下来，我们定义一个函数来训练模型。在每个 epoch 中，模型会遍历训练数据，进行前向传播、计算损失、进行反向传播并更新参数。

```python
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
```

**讲解**：
- `train` 函数通过指定的 epochs 训练模型。每个 epoch 中，模型会遍历训练数据，进行前向传播、计算损失、进行反向传播并更新参数。
- 每个 epoch 结束后，输出当前的平均损失。

### 5. 评估模型

最后，我们定义一个函数来评估模型的准确性。在测试数据上评估模型的表现。

```python
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
```

**讲解**：
- `evaluate` 函数在测试数据上评估模型的准确率。模型进入评估模式，不计算梯度，直接进行预测并计算准确率。
- 输出模型在测试数据上的准确性。

### 完整代码

```python
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
```

这个完整的代码示例展示了如何使用 PyTorch 构建、训练和评估一个简单的全连接神经网络模型。


### 总结

全连接神经网络是最基础的神经网络结构，通过输入层接收数据，通过隐藏层进行特征提取和学习，最后通过输出层给出预测结果。激活函数为网络引入非线性，使其能够学习和处理复杂的模式。通过逐步构建和训练模型，我们可以解决各种数据分类和回归问题。