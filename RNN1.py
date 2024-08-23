import numpy as np

# 定义参数
hidden_size = 4
input_size = 3
sequence_length = 5

# 初始化隐藏状态
h_t = np.zeros(hidden_size)

# 初始化权重矩阵和偏置项
W_hx = np.random.randn(hidden_size, input_size)
W_hh = np.random.randn(hidden_size, hidden_size)
b_h = np.random.randn(hidden_size)

# 定义输入序列
input_sequence = [np.random.randn(input_size) for _ in range(sequence_length)]

# 循环遍历序列中的每一个时间步长
for t in range(sequence_length):
    x_t = input_sequence[t]
    h_t = np.tanh(np.dot(W_hx, x_t) + np.dot(W_hh, h_t) + b_h)
    print(f"时间步长 {t+1} 的隐藏状态: {h_t}")
