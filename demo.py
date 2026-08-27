"""HA-PoolINR command-line demo.

Hide one or more secret media (image / audio / video, mixed allowed) behind a
cover by training a single shared parameter pool, publish only the cover
network, then verify key-based recovery.

Examples:
  # 1 cover image + 1 secret image (paper protocol, 20000 epochs)
  python demo.py --cover data/cover.jpg --secrets data/secret.jpg

  # mixed modalities with your own keys
  python demo.py \
      --cover    data/cover.jpg \
      --secrets  data/secret1.jpg data/secret2.wav data/secret3.mp4 \
      --cover-key     "my-cover-passphrase" \
      --secret-keys   "key-for-secret1" "key-for-secret2" "key-for-secret3"

  # quick smoke test (300 epochs, low quality but fast)
  python demo.py --cover data/cover.jpg --secrets data/secret.jpg --epochs 300
"""
import argparse
from pipeline import run


def main():
    ap = argparse.ArgumentParser(
        description='HA-PoolINR: key-based hiding in a shared INR parameter pool')
    ap.add_argument('--cover', required=True,
                    help='path to the cover media (image/audio/video)')
    ap.add_argument('--secrets', nargs='+', required=True,
                    help='paths to the secret media (any modality, mixed allowed)')
    ap.add_argument('--epochs', type=int, default=20000,
                    help='total training epochs (default 20000, paper protocol)')
    ap.add_argument('--batch-size', type=int, default=10**12,
                    help='samples per step (default: full-batch)')
    ap.add_argument('--lr', type=float, default=3e-3,
                    help='AdamW learning rate (default 3e-3)')
    ap.add_argument('--cover-key', default='COVER_KEY_V1',
                    help='key string for the cover network (USE YOUR OWN)')
    ap.add_argument('--secret-keys', nargs='+', default=None,
                    help='one key string per secret, in order (USE YOUR OWN)')
    ap.add_argument('--out', default='./outputs',
                    help='directory for recovered media')
    ap.add_argument('--artifacts', default='./artifacts',
                    help='directory for published cover model and pool artifacts')
    args = ap.parse_args()

    run(cover_path=args.cover,
        secret_paths=args.secrets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out,
        artifacts_dir=args.artifacts,
        early_stop_loss=None,          # paper protocol: fixed budget
        cover_key=args.cover_key,
        secret_keys=args.secret_keys)


if __name__ == '__main__':
    main()
