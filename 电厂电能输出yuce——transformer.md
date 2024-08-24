## 深度学习实践项目示例——电厂电能输出预测（transformer）

[本项目示例](https://www.kaggle.com/datasets/gauravduttakiit/power-plant-data)选自Kaggle竞赛，旨在通过全连接神经网络（FCNN）模型预测电厂电能输出。以下是详细的代码和解释。

### 项目背景
这些数据点是从一个联合循环电厂在6年（2006-2011）内收集的，当时电厂处于满负荷运行状态。特征包括每小时平均的环境变量——环境温度（AT）、环境压力（AP）、相对湿度（RH）和排气真空（V），用于预测电厂的每小时净电能输出（PE）。

联合循环电厂同时使用燃气轮机和蒸汽轮机，相比传统的简单循环电厂，它能从相同的燃料中生产出多达50％的额外电力。燃气轮机的废热被引导到附近的蒸汽轮机，从而产生额外的电力。

所有联合国成员国都必须向联合国提交有关联合循环电厂的报告。墨西哥的电厂官员正在设计一种方法来预测电厂的每小时净电能输出（PE）。你被任命为这一任务的负责人，请创建一个机器学习模型来有效地解决这个问题。电厂电能输出预测是电力系统运行管理的重要组成部分。通过预测电厂电能输出，可以优化电力系统的调度和运行，提高电力系统的稳定性和效率。

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

使用Transformer模型来解决回归问题（如预测联合循环电厂的每小时净电能输出）需要对传统的Transformer架构进行一些调整。以下是如何使用Transformer模型来解决这个问题的完整代码和详细解释。

### 1. 导入必要的库
首先，我们需要导入相关的Python库，如`pandas`用于数据处理，`numpy`用于数值计算，`matplotlib`用于数据可视化，以及`torch`用于构建和训练Transformer模型。

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

- `pandas` 和 `numpy` 用于处理和操作数据。
- `matplotlib` 用于可视化数据。
- `sklearn` 中的工具用于数据预处理和分割训练集与验证集。
- `torch` 用于构建和训练Transformer模型。

### 2. 加载数据
加载和预处理数据部分与之前使用FCNN的步骤相同。

```python
# 加载训练集和测试集数据
train_data = pd.read_csv('archive/Training_set_ccpp.csv')
test_data = pd.read_csv('archive/Testing_set_ccpp.csv')

# 将训练集的特征和标签分开
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# 将测试集的特征提取出来
X_test = test_data.values

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
```

### 3. 构建Transformer模型
Transformer通常用于处理序列数据，如自然语言处理任务。这里我们将用它来处理固定长度的特征向量。

```python
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_encoder_layers=2, dim_feedforward=256):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # 调整输入形状以匹配 Transformer 的期望输入
        x = self.input_linear(x).unsqueeze(1)  # [batch_size, seq_len=1, d_model]
        x = x.transpose(0, 1)  # [seq_len=1, batch_size, d_model]
        x = self.transformer(x)  # [seq_len=1, batch_size, d_model]
        x = x.mean(dim=0)  # [batch_size, d_model]
        x = self.fc_out(x)  # [batch_size, 1]
        return x

model = TransformerModel(input_dim=X_train.shape[1])
```
 
- `TransformerModel` 类中，我们首先将输入特征映射到 `d_model` 维度，并加上位置编码信息。
- `Transformer` 模块包含多个编码器层。
- 最终使用 `fc_out` 全连接层将编码器的输出映射到目标维度（1）。

### 4. 训练模型
使用PyTorch进行Transformer模型的训练。

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

- `MSELoss` 用于回归问题。
- `Adam` 优化器用于更新Transformer模型参数。

### 5. 使用测试集进行预测
与之前相同，我们将在测试集上评估模型并保存预测结果。

```python
# 进行预测
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)

# 将预测结果转换为numpy数组
predictions_np = predictions.numpy()

# 保存预测结果到CSV文件
submission = pd.DataFrame(data=predictions_np, columns=['PE'])
submission.to_csv('submission.csv', index=False)
```
 
- `eval()` 模式下，我们对测试集数据进行推理，并将结果保存为 `submission.csv`。

### 6. 可视化训练过程
同样，我们可以绘制训练过程中损失的变化曲线。

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

# 将训练集的特征和标签分开
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# 将测试集的特征提取出来
X_test = test_data.values

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# 构建Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_encoder_layers=2, dim_feedforward=256):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # 调整输入形状以匹配 Transformer 的期望输入
        x = self.input_linear(x).unsqueeze(1)  # [batch_size, seq_len=1, d_model]
        x = x.transpose(0, 1)  # [seq_len=1, batch_size, d_model]
        x = self.transformer(x)  # [seq_len=1, batch_size, d_model]
        x = x.mean(dim=0)  # [batch_size, d_model]
        x = self.fc_out(x)  # [batch_size, 1]
        return x

model = TransformerModel(input_dim=X_train.shape[1])

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
train_losses = []
for epoch in range(epochs):
    model.train()
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

# 在测试集上进行预测并保存结果
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)

# 将预测结果转换为numpy数组
predictions_np = predictions.numpy()

# 保存预测结果到CSV文件
submission = pd.DataFrame(data=predictions_np, columns=['PE'])
submission.to_csv('submission.csv', index=False)

# 绘制损失曲线
plt.plot(train_losses, label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

通过这些步骤，您可以使用Transformer模型来预测联合循环电厂的每小时净电能输出。这个代码块将生成一个 `submission.csv` 文件，其中包含模型的预测结果，同时还会生成训练过程中的损失变化曲线。

### 提高精度
要提高Transformer模型的精度，可以调整以下几个关键参数和进行相关操作。这些调整可以帮助模型更好地拟合数据，从而提高预测性能。

### 1. **模型架构参数**
   - **`d_model`（模型维度）**: 这是输入特征映射到的维度。增加 `d_model` 可以提高模型的表达能力，但可能增加过拟合的风险。
   - **`nhead`（注意力头数）**: 这是多头自注意力机制中的头数。增加 `nhead` 可以让模型从不同的子空间中提取信息，但需要平衡计算成本。
   - **`num_encoder_layers`（编码器层数）**: 增加编码器的层数可以增强模型的深度和复杂性，但也会增加计算时间和过拟合的可能性。
   - **`dim_feedforward`（前馈网络的维度）**: 前馈网络的隐藏层维度。增加这个值可以提升模型的非线性能力。

   ```python
   model = TransformerModel(input_dim=X_train.shape[1], d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512)
   ```

### 2. **优化器和学习率**
   - **`learning_rate`（学习率）**: 学习率控制了参数更新的步长。适当地降低学习率（如0.0005或0.0001）可能有助于提高模型的稳定性和精度。
   - **优化器的选择**: 尝试使用不同的优化器，例如 `AdamW` 或 `RMSprop`，以观察对模型性能的影响。

   ```python
   optimizer = optim.AdamW(model.parameters(), lr=0.0005)
   ```

### 3. **训练轮次（Epochs）**
   - **`epochs`（训练轮次）**: 增加训练轮次可以让模型有更多的时间学习数据模式。但要注意的是，如果轮次过多，模型可能会开始过拟合。

   ```python
   epochs = 200  # 可以增加到200甚至更高
   ```

### 4. **批量大小（Batch Size）**
   - **`batch_size`（批量大小）**: 增大或减小批量大小可以影响模型的收敛速度和稳定性。小的批量大小（例如16）可能会更频繁地更新权重，从而增加模型的泛化能力。

   ```python
   train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
   ```

### 5. **正则化技术**
   - **`Dropout`（丢弃法）**: 在模型的某些层中添加Dropout，防止过拟合。Dropout通常在全连接层后或在编码器层之间使用。
   - **权重衰减（Weight Decay）**: 通过在优化器中使用权重衰减，可以减少模型复杂性，从而降低过拟合的风险。

   ```python
   optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
   ```

### 6. **数据增强与预处理**
   - **数据标准化和归一化**: 确保所有输入特征被适当标准化或归一化，以便模型能够更好地学习数据模式。
   - **增加训练数据**: 如果可能，可以通过数据增强或获取更多的数据来增加训练数据的多样性，从而提高模型的泛化能力。

### 7. **调试和验证**
   - **交叉验证**: 使用交叉验证来测试模型的稳健性。
   - **早停法（Early Stopping）**: 在训练过程中监控验证集损失，当验证集损失不再改善时停止训练，以防止过拟合。

   ```python
   early_stopping = EarlyStopping(monitor='val_loss', patience=10)
   ```

通过逐步调整这些参数，可以观察到模型性能的变化，从而找到最适合您数据的模型配置。记得每次只调整一两个参数，观察其效果，然后再继续调整其他参数。