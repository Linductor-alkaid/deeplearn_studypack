import torch
import torch.nn as nn
import numpy as np

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size)
        c_0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# 生成示例数据
data = np.array([[30, 31, 32, 33, 34],
                 [32, 33, 34, 35, 36],
                 [35, 36, 37, 38, 39]],
                 dtype=np.float32)

labels = np.array([35, 37, 40], dtype=np.float32)

data = torch.from_numpy(data).unsqueeze(-1)
labels = torch.from_numpy(labels).unsqueeze(-1)

# 定义超参数
input_size = 1
hidden_size = 10
output_size = 1
num_epochs = 100
learning_rate = 0.01

# 实例化模型
model = WeatherLSTM(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    outputs = model(data)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 进行预测
model.eval()
with torch.no_grad():
    test_input = torch.tensor([[36, 37, 38, 39, 40]], dtype=torch.float32).unsqueeze(-1)
    predicted_temperature = model(test_input)
    print(f"预测的温度: {predicted_temperature.item():.2f}")
