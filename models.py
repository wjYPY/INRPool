import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from kdf import KeyDerivationFunction, derive_seed
from modules import PoolLinear

# ==========================================
# 1. 确定性高频编码层 (Fourier Feature)
# ==========================================
class FourierFeature(nn.Module):
    def __init__(self, in_features: int, mapping_size: int, scale: float = 10.0,
                 key: str = "", device='cuda'):
        """将低维坐标映射为高频周期特征（不占用参数池容量）"""
        super().__init__()
        g = torch.Generator(device=device)
        fourier_seed = derive_seed(key, salt="FOURIER_BASIS")
        g.manual_seed(fourier_seed)
        B = torch.randn((in_features, mapping_size), generator=g, device=device) * scale
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = (2 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ==========================================
# 2. 异构网络封装类 (PoolNetwork)
# ==========================================
class PoolNetwork(nn.Module):
    def __init__(self, key: str, arch: dict, pool_w_size: int, pool_b_size: int,
                 use_fourier=False, fourier_args=None, is_siren=False, device='cuda',
                 # 【新增】Leaky ReLU 的负斜率，0 = 标准 ReLU
                 leaky_slope: float = 0.01):
        super().__init__()
        self.is_siren    = is_siren
        self.leaky_slope = leaky_slope  # 0.01 足够抑制 dying neuron，又不影响正值侧

        # --- [1] Fourier 编码 ---
        self.use_fourier = use_fourier
        if use_fourier and fourier_args is not None:
            fourier_args['key']    = key
            fourier_args['device'] = device
            self.pe = FourierFeature(**fourier_args)

        # --- [2] 全局循环置换索引 ---
        order_W, order_b = [], []
        for name, meta in arch.items():
            n = int(torch.tensor(meta['shape']).prod().item())
            if meta['type'] == 'weight':
                order_W.append((name, n, meta))
            elif meta['type'] == 'bias':
                order_b.append((name, n, meta))

        total_W = sum(n for _, n, _ in order_W)
        total_b = sum(n for _, n, _ in order_b)

        def build_global_indices(p_type, p_size, t_params):
            if t_params == 0: return None
            perm = KeyDerivationFunction.pool_permutation(key, p_size, p_type, device)
            j    = torch.arange(t_params, device=device)
            return perm[j % p_size]

        idx_W = build_global_indices('weight', pool_w_size, total_W)
        idx_b = build_global_indices('bias',   pool_b_size, total_b)

        # --- [3] 空壳 PoolLinear 层 ---
        self.layers = nn.ModuleList()
        cursor_W, cursor_b = 0, 0
        layer_names = [name.replace('_W', '') for name, _, _ in order_W]

        for i, lname in enumerate(layer_names):
            meta_W = arch[f'{lname}_W']
            meta_b = arch[f'{lname}_b']
            nW = int(torch.tensor(meta_W['shape']).prod().item())
            nb = int(torch.tensor(meta_b['shape']).prod().item())

            l_idx_W  = idx_W[cursor_W : cursor_W + nW]
            l_idx_b  = idx_b[cursor_b : cursor_b + nb]
            cursor_W += nW; cursor_b += nb

            l_sign_W = KeyDerivationFunction.sign_mask(key, f'{lname}_W', nW, device)
            l_sign_b = KeyDerivationFunction.sign_mask(key, f'{lname}_b', nb, device)

            out_feat, in_feat = meta_W['shape']
            self.layers.append(PoolLinear(
                in_features=in_feat, out_features=out_feat,
                idx_W=l_idx_W, idx_b=l_idx_b,
                sign_W=l_sign_W, sign_b=l_sign_b,
                s_tau_W=meta_W['s_tau'], s_tau_b=meta_b['s_tau']
            ))

    def forward(self, x: torch.Tensor, P_W: torch.Tensor, P_b: torch.Tensor):
        if self.use_fourier:
            x = self.pe(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, P_W, P_b)
            if i < len(self.layers) - 1:
                if self.is_siren:
                    x = torch.sin(x)
                else:
                    # 【改动】ReLU → Leaky ReLU（slope=0.01）
                    #
                    # 原版 ReLU 的问题：共享池中，一个任务的梯度把某池元素推到负值后，
                    # 另一个任务在同一位置的 ReLU 神经元会永久"死亡"（梯度为 0），
                    # 无法再对那部分池元素产生任何学习信号。
                    #
                    # Leaky ReLU slope=0.01 对正值侧完全等同于 ReLU，
                    # 对负值侧保留 1% 的梯度通道，彻底消除 dying neuron。
                    # 经验上对收敛速度和最终质量都有轻微正向影响。
                    x = F.leaky_relu(x, negative_slope=self.leaky_slope)
        return x