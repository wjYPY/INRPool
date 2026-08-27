import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 idx_W: torch.Tensor, idx_b: torch.Tensor,
                 sign_W: torch.Tensor, sign_b: torch.Tensor,
                 s_tau_W: float = 1.0, s_tau_b: float = 1.0):
        """
        动态空壳线性层：完全由外部参数池和确定的索引驱动。
        【改动】去掉 s_tau 强行缩放，权重 = P_W[idx] * sign
                 s_tau_W / s_tau_b 仍接收（兼容老接口），但不再使用。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('idx_W', idx_W)
        self.register_buffer('idx_b', idx_b)
        self.register_buffer('sign_W', sign_W)
        self.register_buffer('sign_b', sign_b)

        # 保留字段仅用于向后兼容，不再参与 forward
        self.s_tau_W = s_tau_W
        self.s_tau_b = s_tau_b

    def forward(self, x: torch.Tensor, P_W: torch.Tensor, P_b: torch.Tensor) -> torch.Tensor:
        # 直接从池里取，乘 ±1 sign mask；池本身已按 Kaiming 初始化
        weight_flat = P_W[self.idx_W] * self.sign_W
        bias_flat   = P_b[self.idx_b] * self.sign_b

        weight = weight_flat.view(self.out_features, self.in_features)
        bias   = bias_flat.view(self.out_features)
        return F.linear(x, weight, bias)