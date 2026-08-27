import torch
import hashlib

def derive_seed(key: str, salt: str = "") -> int:
    """对 key + salt 做 SHA256,返回前 8 字节作为整数种子。"""
    seed_str = f"{key}||{salt}"
    return int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest()[:16], 16)

class KeyDerivationFunction:
    @staticmethod
    def pool_permutation(key: str, pool_size: int, p_type: str, device='cuda'):
        """
        由 key 生成池的全局置换 π_K(长度 = pool_size)。
        不同参数类型(weight / bias / scale) 用不同 salt,得到独立的置换。
        """
        seed = derive_seed(key, salt=f"POOL_PERM::{p_type}")
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return torch.randperm(pool_size, generator=g, device=device)

    @staticmethod
    def sign_mask(key: str, tensor_name: str, num_params: int, device='cuda'):
        """每个 tensor 自己的 ±1 sign mask(可选,用于增加不可逆性)。"""
        seed = derive_seed(key, salt=f"SIGN::{tensor_name}")
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return torch.randint(0, 2, (num_params,), generator=g, device=device) * 2 - 1