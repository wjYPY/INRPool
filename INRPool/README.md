# HA-PoolINR

**Key-based multi-secret steganography in a shared implicit neural representation (INR) parameter pool.**

HA-PoolINR hides an arbitrary number of secret media (images / audio / video, mixed modalities) behind a single cover by overfitting **one global shared parameter pool** (~526 K trainable values). Every task network — the cover and each secret — is *assembled from the same pool* using a key derived from a passphrase. Only the cover network is ever transmitted; it looks like an ordinary INR. A receiver with the correct keys decodes the pool back out of the published cover network and re-assembles the secret networks.

## How it works

1. **Shared pool** — a single flat tensor `P` (524,544 weights + 1,281 biases, Kaiming-initialized) is the *only* trainable object in the system.
2. **Key derivation (KDF)** — `seed = SHA256(key || salt)`. From one key, three independent oracles are derived:
   - a **pool permutation** `π` (position domain: which pool entry fills which network slot),
   - a **±1 sign mask** `s` (value domain: the polarity each entry enters with),
   - a **Fourier basis** `B` (input domain: how coordinates are encoded).
3. **Keyed assembly** — network slot *j* of a task reads `w_j = s_j · P[π(j mod N)]`. Each network reuses essentially the whole pool; only the arrangement and polarity differ per key.
4. **Joint training** — all task networks are trained simultaneously; the joint loss is the **equal-weight sum** of the per-task MSE losses and only the pool receives gradients (each pool entry is updated by the sum of its signed contributions from every network that uses it). The pool at the end of the fixed budget is published.
5. **Publish** — only the cover network is exported as a normal-looking INR state dict (`cover_published.pth`, ~2 MB).
6. **Decode** — a receiver holding the cover key inverts the sign mask (±1 is self-inverse) and scatters the published values back through the permutation to recover the pool, then re-assembles any secret network from it with the corresponding secret key.

Security intuition: a wrong key yields a wrong permutation *and* wrong signs, so the assembled network outputs noise-level reconstructions (~6 dB PSNR). The sign mask additionally erases the exact-equality fingerprint that wrap-around collisions (slot *j* vs *j+N*) would otherwise leave in the published weights.

## Installation

```bash
conda create -n hapoolinr python=3.10 -y
conda activate hapoolinr
pip install torch torchvision torchaudio   # CUDA build recommended
pip install pytorch-msssim                 # standard SSIM (11x11 Gaussian window)
pip install lpips                          # optional, video LPIPS metric only
```

## Usage

Put any media files into `data/` (`.jpg/.png`, `.wav/.mp3`, `.mp4`, ...). Default modality configs: images 256×256, audio 3 s @ 8 kHz, video 8 frames @ 128×128 (see `configs.py`).

```bash
# 1 cover image + 1 secret image, paper protocol (20000 epochs)
python demo.py --cover data/cover.jpg --secrets data/secret.jpg

# mixed modalities with custom keys
python demo.py \
    --cover       data/cover.jpg \
    --secrets     data/secret1.jpg data/secret2.wav data/secret3.mp4 \
    --cover-key   "my-cover-passphrase" \
    --secret-keys "key-1" "key-2" "key-3"
```

Outputs:
- `artifacts/cover_published.pth` — the **public** cover network (the only thing you transmit),
- `artifacts/real_pool_GT.pth` — sender-side ground-truth pool (decode-accuracy check only),
- `artifacts/trained_model.pth` — full dump (pool + networks + config) for further analysis,
- `outputs/recovered_*.jpg|wav|mp4` — reconstructions evaluated from the *recovered* pool,
- console metrics for cover/secret quality and a **wrong-key attack** baseline.

## Reference numbers

Single cover image + single secret image, 256×256, 20000 epochs, RTX 4090 (~10 min):

| Quantity | PSNR |
|---|---|
| Cover reconstruction | ~61.6 dB |
| Secret reconstruction (correct key) | ~58.2 dB |
| Secret reconstruction (wrong key) | ~6.2 dB |

Capacity scaling (cover PSNR / mean secret PSNR, same protocol, Div2k 0801 as cover):

| #secrets | 2 | 5 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|
| cover | 50.6 | 38.4 | 33.6 | 32.5 | 29.7 | 27.4 |
| secret | 52.7 | 38.2 | 32.2 | 30.9 | 28.3 | 25.8 |

Quality degrades gracefully with the number of secrets — this is the inherent trade-off of permutation-level parameter sharing (every pool value must simultaneously serve every network), which buys key security and statistical undetectability in return.

## Repository layout

| File | Role |
|---|---|
| `pool.py` | Global shared parameter pool (`TypedParamPool`), Kaiming init variants |
| `kdf.py` | Key derivation: SHA256 → permutation `π`, ±1 sign masks, Fourier seeds |
| `modules.py` | `PoolLinear` — the weight-free layer: `W = P[idx] ⊙ sign` |
| `models.py` | `PoolNetwork` (key-seeded Fourier features + stacked `PoolLinear`) |
| `filler.py` | Publish (pool → cover tensors) and decode (cover tensors → pool) |
| `configs.py` | Modality configs, architecture builder, pool capacity constants |
| `data_io.py` | Media loading/saving for image / audio / video |
| `metrics.py` | PSNR / SSIM / MAE / RMSE / SNR / APD / LPIPS evaluators |
| `pipeline.py` | Full train → publish → decode → evaluate pipeline (`run()`) |
| `demo.py` | Command-line interface |

## Notes

- **Keys are passphrases.** Change `--cover-key` / `--secret-keys` for any real use; the defaults are placeholders.
- The published cover file is a plain state dict of tensors — bit-for-bit indistinguishable in form from an ordinary trained INR.
- Training follows the paper protocol: pure MSE losses combined by an equal-weight sum (no task-count averaging), gradient-norm clipping 1.0, fixed 20000-epoch budget, final pool published directly.

## Citation

If you use this code, please cite:

```bibtex
@article{yang2026hapoolinr,
  title  = {HA-PoolINR: Key-Based Multi-Secret Steganography via a Shared Implicit Neural Representation Parameter Pool},
  author = {Yang, Pengyuan},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
