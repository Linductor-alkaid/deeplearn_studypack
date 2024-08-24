## 深度学习实践项目示例——电厂电能输出预测（FCNN）

[本项目示例](https://www.kaggle.com/datasets/gauravduttakiit/power-plant-data)选自Kaggle竞赛，旨在通过全连接神经网络（FCNN）模型预测电厂电能输出。以下是详细的代码和解释。

### 项目背景
这些数据点是从一个联合循环电厂在6年（2006-2011）内收集的，当时电厂处于满负荷运行状态。特征包括每小时平均的环境变量——环境温度（AT）、环境压力（AP）、相对湿度（RH）和排气真空（V），用于预测电厂的每小时净电能输出（PE）。

联合循环电厂同时使用燃气轮机和蒸汽轮机，相比传统的简单循环电厂，它能从相同的燃料中生产出多达50％的额外电力。燃气轮机的废热被引导到附近的蒸汽轮机，从而产生额外的电力。

所有联合国成员国都必须向联合国提交有关联合循环电厂的报告。墨西哥的电厂官员正在设计一种方法来预测电厂的每小时净电能输出（PE）。你被任命为这一任务的负责人，请创建一个机器学习模型来有效地解决这个问题。电厂电能输出预测是电力系统运行管理的重要组成部分。通过预测电厂电能输出，可以优化电力系统的调度和运行，提高电力系统的稳定性和效率。本项目使用全连接神经网络（FCNN）模型来预测电厂电能输出。

### 数据集
数据集包含6年的每小时数据，每行数据包括4个特征（环境温度、环境压力、相对湿度、排气真空）和1个标签（电厂电能输出）。数据集分为训练集和测试集，训练集用于训练模型，测试集用于评估模型的性能。

数据集的下载可以去官网下载，不过kaggle需要注册账号，这里我提供一下数据集的[下载链接](https://download.csdn.net/download/qq_41065669/89670592)

文件结构如下：

```
archive/
    Training_set_ccpp.csv
    Testing_set_ccpp.csv
FCNN.py
```

### 模型选择

下面是一个使用FCNN（全连接神经网络）解决该Kaggle题目的完整代码和详细解释：

### 1. 导入必要的库
首先，我们需要导入相关的Python库，如`pandas`用于数据处理，`numpy`用于数值计算，`matplotlib`用于数据可视化，以及`sklearn`和`torch`用于机器学习和深度学习。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```
- `pandas`和`numpy`用于处理和操作数据。
- `matplotlib`用于可视化数据。
- `sklearn`中的工具用于数据预处理和分割训练集与验证集
- `torch` 是 PyTorch 的核心库。
- `torch.nn` 包含了用于构建神经网络的模块。
- `torch.optim` 包含了用于优化的算法。
- `DataLoader` 和 `TensorDataset` 用于处理数据加载。

### 2. 加载数据
我们需要从CSV文件中加载训练数据和测试数据。

```python
train_data = pd.read_csv('archive/Training_set_ccpp.csv')
test_data = pd.read_csv('archive/Testing_set_ccpp.csv')
```
- 使用`pandas`的`read_csv`函数从指定路径加载CSV文件。

### 3. 数据预处理
我们需要将数据划分为特征（输入）和标签（输出），并标准化数据以提高模型训练的效果。

```python
# 将特征和标签分开
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.values
y_test = test_data.iloc[:, -1].values

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
```

- `iloc`用于根据位置选择数据，`:-1`选择所有行的所有列，除了最后一列（特征部分），`-1`选择最后一列（标签部分）。
- `StandardScaler`用于标准化数据，使每个特征的均值为0，方差为1，提升模型训练效果。
- `torch.tensor` 用于将numpy数组转换为PyTorch张量。
- `TensorDataset` 将特征和标签打包在一起，`DataLoader` 用于按批次加载数据。

### 4. 构建FCNN模型
我们使用PyTorch定义一个简单的全连接神经网络模型。

```python
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = FCNN(input_dim=X_train.shape[1])
```


- `FCNN` 类继承了 `nn.Module`，定义了一个包含两层隐藏层和一层输出层的全连接神经网络。
- `forward` 函数定义了数据如何通过网络层的流程。

### 5. 训练模型
使用PyTorch进行模型训练。

```python
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
```

**解释**:  
- `MSELoss` 是用于回归问题的均方误差损失函数。
- `Adam` 优化器用于更新模型参数。
- 每个 epoch，我们在训练数据上训练模型，并每10个epoch打印一次损失。

### 6. 评估模型
使用测试数据评估模型的性能。

```python
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
```

**解释**:  
- `model.eval()` 将模型设置为评估模式。
- `torch.no_grad()` 停止跟踪梯度，进行推理。

### 7. 可视化训练过程
如果需要，可以手动记录并绘制损失曲线。

```python
# 手动记录训练过程中的损失以绘制
train_losses = []

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

# 绘制损失曲线
plt.plot(train_losses, label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**解释**:  
- `train_losses` 记录每个epoch的平均损失。
- 使用`matplotlib`绘制损失曲线。

### 完整代码块

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 加载训练集和测试集数据
train_data = pd.read_csv('archive/Training_set_ccpp.csv')
test_data = pd.read_csv('archive/Testing_set_ccpp.csv')

# 将特征和标签分开
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.values
y_test = test_data.iloc[:, -1].values

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# 构建FCNN模型
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = FCNN(input_dim=X_train.shape[1])

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# 绘制训练和验证损失
train_losses = []

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

# 绘制损失曲线
plt.plot(train_losses, label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt

.legend()
plt.show()
```

通过这个步骤，您可以使用全连接神经网络来预测联合循环电厂的每小时净电能输出。