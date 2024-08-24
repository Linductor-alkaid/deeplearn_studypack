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