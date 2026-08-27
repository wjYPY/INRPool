import torch
import torchvision.io as tvio
import torchaudio
import os
def check_path(path):
    if not os.path.exists(path): raise FileNotFoundError(f"找不到文件: {path}")
# ---------- 加载 ----------
def load_image(path, resolution, device='cuda'):
    img = tvio.read_image(path).float() / 255.0  # [C,H,W], C 可能是 1/3/4
    if img.shape[0] == 1: img = img.repeat(3, 1, 1)
    if img.shape[0] == 4: img = img[:3]
    img = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=resolution, mode='bilinear', align_corners=False
    ).squeeze(0)  # [3,H,W]
    H, W = resolution
    y, x = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing='ij')
    coords = torch.stack([x, y], dim=-1).view(-1, 2).to(device)
    rgb = img.permute(1,2,0).view(-1, 3).to(device)
    return coords, rgb, {'resolution': resolution}

def load_audio(path, duration, sample_rate, device='cuda'):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    length = int(sample_rate * duration)
    if wav.shape[1] < length:
        wav = torch.nn.functional.pad(wav, (0, length - wav.shape[1]))
    wav = wav[0, :length].unsqueeze(-1)  # [N,1]
    peak = wav.abs().max().clamp(min=1e-8)
    wav = wav / peak
    
    wav = wav.to(device)
    t = torch.linspace(-1, 1, length).unsqueeze(-1).to(device)
    return t, wav, {'sample_rate': sample_rate, 'length': length}

# --- data_io.py 中的 load_video 替换 ---

def load_video(path, num_frames, resolution, device='cuda',
               start_pts: float = 0.0, end_pts: float = 2.0):
    """
    读取 [start_pts, end_pts] 秒范围内的视频，均匀采 num_frames 帧。
    额外在 meta 里保存 frame_times（每帧对应的真实秒数），
    供可视化和评测按时间对齐。
    """
    check_path(path)
    frames, _, info = tvio.read_video(
        path, start_pts=start_pts, end_pts=end_pts, pts_unit='sec'
    )
    frames = frames.float() / 255.0  # [T,H,W,3]
    T_raw = frames.shape[0]
    duration = max(end_pts - start_pts, 1e-6)

    # 均匀采样索引 & 对应时间戳（秒，绝对时间）
    if T_raw >= num_frames:
        idx = torch.linspace(0, T_raw - 1, num_frames).long()
        frames = frames[idx]
        # 每帧在原视频中的时间戳
        src_fps = info.get('video_fps', T_raw / duration)
        frame_times = (start_pts + idx.float() / max(src_fps, 1e-6)).tolist()
    else:
        # 尾部补最后一帧
        pad_n = num_frames - T_raw
        frames = torch.cat(
            [frames, frames[-1:].repeat(pad_n, 1, 1, 1)], dim=0
        )
        # 已有帧按原 fps 计算，其余用最后一帧的时间戳
        src_fps = info.get('video_fps', T_raw / duration) if T_raw > 0 else 1.0
        base = [start_pts + i / max(src_fps, 1e-6) for i in range(T_raw)]
        last_t = base[-1] if base else start_pts
        frame_times = base + [last_t] * pad_n

    frames = torch.nn.functional.interpolate(
        frames.permute(0, 3, 1, 2), size=resolution,
        mode='bilinear', align_corners=False
    )
    T_, _, H, W = frames.shape
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, T_),
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    coords = torch.stack([x, y, z], dim=-1).view(-1, 3).to(device)
    target = frames.permute(0, 2, 3, 1).reshape(-1, 3).to(device)

    meta = {
        'num_frames': T_,
        'resolution': resolution,
        'start_pts':  float(start_pts),
        'end_pts':    float(end_pts),
        'duration':   float(duration),
        'frame_times': [float(t) for t in frame_times],  # 长度 = num_frames
        'save_fps':   float(num_frames / duration),      # 我们保存时使用的 fps
    }
    return coords, target, meta

LOADERS = {'image': load_image, 'audio': load_audio, 'video': load_video}

def load_data(modality, path, cfg, device='cuda'):
    if modality == 'image':
        return load_image(path, cfg['resolution'], device)
    if modality == 'audio':
        return load_audio(path, cfg['duration'], cfg['sample_rate'], device)
    if modality == 'video':
        return load_video(path, cfg['num_frames'], cfg['resolution'], device)

# ---------- 写出 ----------
def save_image(pred_rgb, meta, out_path):
    H, W = meta['resolution']
    img = pred_rgb.clamp(0, 1).view(H, W, 3).permute(2, 0, 1).cpu()
    tvio.write_jpeg((img * 255).byte(), out_path, quality=95)

def save_audio(pred_amp, meta, out_path):
    wav = pred_amp.squeeze(-1).clamp(-1, 1).cpu().unsqueeze(0)
    torchaudio.save(out_path, wav, sample_rate=meta['sample_rate'])

def save_video(pred_rgb, meta, out_path):
    """
    保存视频。fps 采用 num_frames / duration，
    使得保存后的 mp4 时间轴与 GT 完全对齐。
    """
    T = meta['num_frames']
    H, W = meta['resolution']
    fps = meta.get('save_fps', T / max(meta.get('duration', 1.0), 1e-6))
    vid = pred_rgb.clamp(0, 1).view(T, H, W, 3).cpu()
    vid_u8 = (vid * 255).byte()
    tvio.write_video(out_path, vid_u8, fps=float(fps))

SAVERS = {'image': save_image, 'audio': save_audio, 'video': save_video}