# metrics.py  ——  学术期刊标准版（音频 MAE 已修正到 0-255 域）
# 图像：PSNR, SSIM, MAE, RMSE       （像素值 0-255）
# 视频：PSNR, SSIM, APD, LPIPS      （像素值 0-255；LPIPS 在 [-1,1] 上算）
# 音频：MAE, SNR                    （MAE 在 0-255 域；SNR 在波形域，dB）
import math
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
try:
    from pytorch_msssim import ssim as _pmsssim_ssim
    _HAS_PMSSSIM = True
except Exception:
    _HAS_PMSSSIM = False
    import warnings
    warnings.warn(
        "[metrics] 未安装 pytorch_msssim，回退到 3x3 uniform 近似 SSIM。"
        "  pip install pytorch-msssim  推荐安装以对齐主流论文口径。"
    )
# =====================================================
# 通用工具
# =====================================================
def _to_uint_like(t, p):
    """[0,1] -> [0,255] (float, 不取整以保留精度)"""
    return t.float().clamp(0, 1) * 255.0, p.float().clamp(0, 1) * 255.0


def _psnr_255(true_255, pred_255):
    mse = F.mse_loss(pred_255, true_255).item()
    return 10.0 * math.log10((255.0 ** 2) / (mse + 1e-12)), mse


def _mae_rmse_255(true_255, pred_255):
    diff = (pred_255 - true_255).abs()
    return diff.mean().item(), math.sqrt((diff ** 2).mean().item())


def _ssim(true_img_hw3, pred_img_hw3):
    """
    单帧 SSIM。
    - 若安装了 pytorch_msssim：使用 11x11 高斯窗、data_range=1.0（标准口径）。
    - 否则回退到之前的 3x3 uniform 近似。
    输入: [H,W,3]，值域 [0,1]
    """
    x = true_img_hw3.permute(2, 0, 1).unsqueeze(0).float().clamp(0, 1)
    y = pred_img_hw3.permute(2, 0, 1).unsqueeze(0).float().clamp(0, 1)

    if _HAS_PMSSSIM:
        # pytorch_msssim 默认 win_size=11, sigma=1.5，高斯窗
        val = _pmsssim_ssim(
            x, y,
            data_range=1.0,
            size_average=True,
            win_size=11,
        )
        return float(val.item())

    # ---- fallback: 之前的 3x3 uniform 近似 ----
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sx  = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sy  = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / \
        ((mu_x ** 2 + mu_y ** 2 + C1) * (sx + sy + C2))
    return float(m.mean().item())


# =====================================================
# 图像评测：PSNR, SSIM, MAE, RMSE  (0-255)
# =====================================================
def eval_image(true_rgb, pred_rgb, meta):
    pred_rgb = pred_rgb.clamp(0, 1)
    H, W = meta['resolution']

    t255, p255 = _to_uint_like(true_rgb, pred_rgb)
    psnr, _ = _psnr_255(t255, p255)
    mae, rmse = _mae_rmse_255(t255, p255)
    ssim = _ssim(true_rgb.view(H, W, 3), pred_rgb.view(H, W, 3))

    return {
        'PSNR(dB)': psnr,
        'SSIM'    : ssim,
        'MAE'     : mae,    # 0-255
        'RMSE'    : rmse,   # 0-255
    }


# =====================================================
# 视频评测：PSNR, SSIM, APD, LPIPS  (0-255；LPIPS 内部映射到 [-1,1])
# =====================================================
_LPIPS_NET = None


def _get_lpips(device):
    global _LPIPS_NET
    if _LPIPS_NET is not None:
        return _LPIPS_NET
    try:
        import lpips  # type: ignore
        _LPIPS_NET = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
        for p in _LPIPS_NET.parameters():
            p.requires_grad_(False)
        return _LPIPS_NET
    except Exception as e:
        print(f"  [LPIPS] 未启用 (原因: {e}). 跳过 LPIPS 计算。")
        _LPIPS_NET = False
        return None


@torch.no_grad()
def _lpips_video(true_thw3, pred_thw3):
    device = true_thw3.device
    net = _get_lpips(device)
    if net is None or net is False:
        return float('nan')

    T = true_thw3.shape[0]
    x = true_thw3.permute(0, 3, 1, 2).float() * 2.0 - 1.0
    y = pred_thw3.permute(0, 3, 1, 2).float() * 2.0 - 1.0
    vals = []
    bs = 8
    for i in range(0, T, bs):
        vals.append(net(x[i:i + bs], y[i:i + bs]).flatten())
    return torch.cat(vals).mean().item()


def eval_video(true_rgb, pred_rgb, meta):
    pred_rgb = pred_rgb.clamp(0, 1)
    T = meta['num_frames']
    H, W = meta['resolution']

    t = true_rgb.view(T, H, W, 3)
    p = pred_rgb.view(T, H, W, 3)

    t255, p255 = _to_uint_like(t, p)
    psnr, _ = _psnr_255(t255, p255)
    apd = (p255 - t255).abs().mean().item()  # Average Pixel Discrepancy (0-255)

    ssim = sum(_ssim(t[i], p[i]) for i in range(T)) / T
    lpips_val = _lpips_video(t, p)

    out = {'PSNR(dB)': psnr, 'SSIM': ssim, 'APD': apd}
    if not (isinstance(lpips_val, float) and math.isnan(lpips_val)):
        out['LPIPS'] = lpips_val
    return out


import math
import torch

# =====================================================
# 音频评测：MAE (0-255域), SNR (dB)
# 严格遵循论文要求："All data are represented within the range of 0 to 255."
# =====================================================
def eval_audio(true_wav, pred_wav, meta):
    # 1. 扁平化，并确保波形在 [-1, 1] 范围内
    x = true_wav.flatten().float().clamp(-1.0, 1.0)
    y = pred_wav.flatten().float().clamp(-1.0, 1.0)

    # 2. 核心：将真实波形和预测波形严格映射到 [0, 255] 范围
    # 公式：x_norm ∈ [-1, 1] -> (x_norm + 1) * 127.5 ∈ [0, 255]
    x_255 = (x + 1.0) * 127.5
    y_255 = (y + 1.0) * 127.5

    # 3. 计算 0-255 范围下的 MAE
    mae_255 = (y_255 - x_255).abs().mean().item()

    # 4. 计算 SNR (dB)
    # 注意：SNR是信号能量与噪声能量的比值（对数域）。
    # 物理意义上，SNR 应该在零均值的原始波形 [-1, 1] 上计算。
    sig_e = x.pow(2).sum().item()
    err_e = (x - y).pow(2).sum().item()
    snr = 10.0 * math.log10((sig_e + 1e-12) / (err_e + 1e-12))

    return {
        'MAE'    : mae_255,
        'SNR(dB)': snr,
    }


# =====================================================
# 注册表 + 打印
# =====================================================
EVALUATORS = {'image': eval_image, 'audio': eval_audio, 'video': eval_video}


def pretty_print(name, metrics_dict):
    print(f"\n【{name}】")
    for k, v in metrics_dict.items():
        # 根据你的论文设定，分门别类保留对应格式
        if k in ('MAE', 'RMSE', 'APD'):
            print(f"  - {k:<10}: {v:.4f}")
        elif k in ('PSNR(dB)', 'SNR(dB)'):
            print(f"  - {k:<10}: {v:.4f}")
        elif k in ('SSIM', 'LPIPS'):
            print(f"  - {k:<10}: {v:.6f}")
        else:
            print(f"  - {k:<10}: {v:.6f}")