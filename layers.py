import torch
import torch.nn as nn

# -------------------------
# 多层感知机（加入 LayerNorm 以增强小 batch 稳定性）
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)  
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)   
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x


# -------------------------
# 残差层 ResidualLayer（增加 LayerNorm + 残差归一）
# -------------------------
class ResidualLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        """
        残差层：输入 -> hidden -> 输出，再与输入相加
        增强稳定性，特别适合 batch_size=1 训练
        """
        super(ResidualLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim)
        )
        self.norm = nn.LayerNorm(in_dim)   

    def forward(self, x):
        return self.norm(x + self.fc(x))
