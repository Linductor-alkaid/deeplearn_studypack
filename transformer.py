import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        # 初始化线性变换矩阵
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 线性变换并拆分成多头
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        
        # 拼接多头输出
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.num_heads)
        
        # 最终线性变换
        output = self.out(context)
        return output

# 示例数据输入
input_seq = torch.rand(2, 5, 8)  # batch_size = 2
multi_head_attn = MultiHeadAttention(d_model=8, num_heads=2)
output = multi_head_attn(input_seq)
print(f"多头注意力层的输出: \n{output}")

