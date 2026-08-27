import torch
from pool import TypedParamPool
from kdf import KeyDerivationFunction

class FillerProtocol:
    def __init__(self, device='cuda'):
        self.device = device

    def _build_global_indices(self, key, p_type, pool_size, total_params):
        perm = KeyDerivationFunction.pool_permutation(key, pool_size, p_type, self.device)
        j = torch.arange(total_params, device=self.device)
        return perm[j % pool_size]

    def fill_model(self, pool: TypedParamPool, key: str, arch: dict):
        order_W, order_b = [], []
        for name, meta in arch.items():
            n = int(torch.tensor(meta['shape']).prod().item())
            if meta['type'] == 'weight':
                order_W.append((name, n, meta))
            elif meta['type'] == 'bias':
                order_b.append((name, n, meta))
            else:
                raise ValueError(meta['type'])

        total_W = sum(n for _, n, _ in order_W)
        total_b = sum(n for _, n, _ in order_b)
        idx_W = self._build_global_indices(key, 'weight', pool.n_w, total_W) if total_W else None
        idx_b = self._build_global_indices(key, 'bias',   pool.n_b, total_b) if total_b else None

        out = {}
        cursor_W = 0
        for name, n, meta in order_W:
            ind = idx_W[cursor_W:cursor_W + n]; cursor_W += n
            signs = KeyDerivationFunction.sign_mask(key, name, n, self.device)
            vals = pool.P_W[ind] * signs                # 不再乘 s_tau
            out[name] = {'data': vals.view(meta['shape']),
                         'type': 'weight', 's_tau': meta['s_tau'], 'num': n}
        cursor_b = 0
        for name, n, meta in order_b:
            ind = idx_b[cursor_b:cursor_b + n]; cursor_b += n
            signs = KeyDerivationFunction.sign_mask(key, name, n, self.device)
            vals = pool.P_b[ind] * signs                # 不再乘 s_tau
            out[name] = {'data': vals.view(meta['shape']),
                         'type': 'bias', 's_tau': meta['s_tau'], 'num': n}
        return out

    def decode_to_pool(self, tensors: dict, key: str, arch: dict,
                       pool_w_size: int, pool_b_size: int):
        order_W, order_b = [], []
        for name, meta in arch.items():
            n = int(torch.tensor(meta['shape']).prod().item())
            if meta['type'] == 'weight':
                order_W.append((name, n))
            else:
                order_b.append((name, n))

        total_W = sum(n for _, n in order_W)
        total_b = sum(n for _, n in order_b)
        idx_W = self._build_global_indices(key, 'weight', pool_w_size, total_W) if total_W else None
        idx_b = self._build_global_indices(key, 'bias',   pool_b_size, total_b) if total_b else None

        rec_W = torch.zeros(pool_w_size, device=self.device)
        cnt_W = torch.zeros(pool_w_size, device=self.device)
        rec_b = torch.zeros(pool_b_size, device=self.device)
        cnt_b = torch.zeros(pool_b_size, device=self.device)

        cursor = 0
        for name, n in order_W:
            meta = tensors[name]
            flat = meta['data'].flatten()
            signs = KeyDerivationFunction.sign_mask(key, name, n, self.device)
            orig = flat * signs                           # 不再除 s_tau
            ind = idx_W[cursor:cursor + n]; cursor += n
            rec_W.scatter_add_(0, ind, orig)
            cnt_W.scatter_add_(0, ind, torch.ones_like(ind, dtype=torch.float))

        cursor = 0
        for name, n in order_b:
            meta = tensors[name]
            flat = meta['data'].flatten()
            signs = KeyDerivationFunction.sign_mask(key, name, n, self.device)
            orig = flat * signs                           # 不再除 s_tau
            ind = idx_b[cursor:cursor + n]; cursor += n
            rec_b.scatter_add_(0, ind, orig)
            cnt_b.scatter_add_(0, ind, torch.ones_like(ind, dtype=torch.float))

        rec_W = rec_W / cnt_W.clamp(min=1)
        rec_b = rec_b / cnt_b.clamp(min=1)
        return rec_W, rec_b, cnt_W, cnt_b