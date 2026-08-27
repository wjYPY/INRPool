import os
import math

POOL_W_CAPACITY = 524544 #N_W=525,056, N_b=1,283，524544 ，1281
POOL_B_CAPACITY = 1281

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
AUDIO_EXT = {'.wav', '.mp3', '.flac', '.ogg'}
VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv'}

def detect_modality(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT: return 'image'
    if ext in AUDIO_EXT: return 'audio'
    if ext in VIDEO_EXT: return 'video'
    raise ValueError(f"不支持的文件类型: {path}")

# =====================================================
# 【核心改动】宽度优先 (Matryoshka 过参数化原则)
#   - hidden: 加宽到 512
#   - depth : 减到 3
#   - is_siren: 全部默认关闭（保留字段方便后续开启）
# =====================================================

IMAGE_CFG = {
    'resolution': (256, 256),
    'in_dim': 2, 'out_dim': 3,
    'hidden': 256, 'depth': 6,
    'use_fourier': True, 'is_siren': False,
    'fourier_mapping': 512, 'fourier_scale': 10.0,
    'omega_0': 30.0,
}

AUDIO_CFG = {
    'duration': 3, 'sample_rate': 8000,
    'in_dim': 1, 'out_dim': 1,
    'hidden': 256, 'depth': 6,
    'use_fourier': True, 'is_siren': False,
    'fourier_mapping': 512,    # 256 → 512：更密集的频率基覆盖
    'fourier_scale': 300.0,    # 50 → 300：覆盖到接近 Nyquist (4 kHz)
    'omega_0': 30.0,
}

VIDEO_CFG = {
    'num_frames': 8, 'resolution': (128, 128),
    'in_dim': 3, 'out_dim': 3,
    'hidden':256, 'depth': 6,
    'use_fourier': True, 'is_siren': False,
    'fourier_mapping': 512, 'fourier_scale': 10.0,
    'omega_0': 30.0,
}

MODALITY_CFG = {'image': IMAGE_CFG, 'audio': AUDIO_CFG, 'video': VIDEO_CFG}

def build_arch(cfg: dict) -> dict:
    """
    【改动】s_tau 字段保留以兼容老协议，但统一固定为 1.0；
    PoolLinear/Filler 内部已经不再使用 s_tau 做缩放。
    SIREN 分支保留：仅当 cfg['is_siren']=True 时，s_tau 才会按 SIREN 论文设置；
    默认 ReLU 路径下，所有 s_tau == 1.0。
    """
    in_dim, out_dim = cfg['in_dim'], cfg['out_dim']
    H, L = cfg['hidden'], cfg['depth']
    is_siren = cfg['is_siren']
    omega = cfg.get('omega_0', 30.0)

    first_in = 2 * cfg['fourier_mapping'] if cfg['use_fourier'] else in_dim
    dims = [first_in] + [H] * (L - 1) + [out_dim]
    arch = {}

    for i in range(L):
        fan_in, fan_out = dims[i], dims[i+1]
        name = f'layer{i+1}'

        if is_siren:
            # 仅供未来开启 SIREN 时使用；当前默认走 else 分支
            if i == 0:
                s_W = omega * math.sqrt(2 / fan_in) if cfg.get('use_fourier') else omega
                s_b = 1.0
            elif i == L - 1:
                s_W, s_b = math.sqrt(2 / fan_in), 0.01
            else:
                s_W, s_b = omega * math.sqrt(2 / fan_in), 0.01
        else:
            # ReLU MLP：池本身已经 Kaiming，所有层 s_tau 统一为 1
            s_W, s_b = 1.0, 1.0

        arch[f'{name}_W'] = {'shape': (fan_out, fan_in), 'type': 'weight', 's_tau': s_W}
        arch[f'{name}_b'] = {'shape': (fan_out,),        'type': 'bias',   's_tau': s_b}
    return arch

def fourier_args_from_cfg(cfg: dict):
    if not cfg['use_fourier']: return None
    return {'in_features': cfg['in_dim'],
            'mapping_size': cfg['fourier_mapping'],
            'scale': cfg['fourier_scale']}

def _prod(shape):
    p = 1
    for s in shape: p *= s
    return p

def count_params(arch: dict):
    nW = sum(_prod(m['shape']) for m in arch.values() if m['type']=='weight')
    nb = sum(_prod(m['shape']) for m in arch.values() if m['type']=='bias')
    return nW, nb