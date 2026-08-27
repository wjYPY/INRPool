import math
import torch

class TypedParamPool:
    def __init__(self, n_w: int, n_b: int, device='cuda',
                 ref_fan_in: int = 512, init_scheme: str = 'kaiming_normal', 
                 omega_0: float = 30.0):
        """
        全局参数池 (支持多初始化消融)
        :param init_scheme: 初始化策略名称
        :param omega_0: SIREN 专用的频率缩放因子
        """
        self.n_w = n_w
        self.n_b = n_b
        self.device = device

        # ================= 权重初始化消融 =================
        if init_scheme == 'kaiming_normal':
            # 默认方案：适用于 ReLU / LeakyReLU
            std_w = math.sqrt(2.0 / ref_fan_in)
            self.P_W = torch.randn(n_w, device=device, dtype=torch.float32) * std_w
            
        elif init_scheme == 'kaiming_uniform':
            bound = math.sqrt(6.0 / ref_fan_in)
            self.P_W = (torch.rand(n_w, device=device, dtype=torch.float32) * 2 - 1) * bound
            
        elif init_scheme == 'xavier_normal':
            # 假设 fan_in ≈ fan_out 时的 Xavier 近似
            std_w = math.sqrt(2.0 / (ref_fan_in + ref_fan_in))
            self.P_W = torch.randn(n_w, device=device, dtype=torch.float32) * std_w
            
        elif init_scheme == 'siren':
            # 适用于正弦激活函数
            bound = math.sqrt(6.0 / ref_fan_in) / omega_0
            self.P_W = (torch.rand(n_w, device=device, dtype=torch.float32) * 2 - 1) * bound
            
        elif init_scheme == 'normal_01':
            # 简单的高斯白噪声 (方差0.1)
            self.P_W = torch.randn(n_w, device=device, dtype=torch.float32) * 0.1
            
        elif init_scheme == 'zeros':
            # 死亡对照组：验证全零初始化会导致对称性破坏和梯度消失
            self.P_W = torch.zeros(n_w, device=device, dtype=torch.float32)
            
        else:
            raise ValueError(f"Unknown init scheme: {init_scheme}")

        # ================= 偏置初始化消融 =================
        if init_scheme == 'zeros':
            self.P_b = torch.zeros(n_b, device=device, dtype=torch.float32)
        else:
            # 加入 1e-3 的微小噪声打破严格对称性
            self.P_b = torch.randn(n_b, device=device, dtype=torch.float32) * 1e-3

    def get_pool(self, p_type: str):
        if p_type == 'weight':
            return self.P_W
        elif p_type == 'bias':
            return self.P_b
        raise ValueError(f"Unknown pool type: {p_type}")