#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_FILES = [
    'rdppimi-s4-3b-weighted-pair-fold1-best.pt',
    'rdppimi-s4-3b-weighted-pair-fold2-best.pt',
    'rdppimi-s4-3b-weighted-pair-fold3-best.pt',
    'rdppimi-s4-3b-weighted-pair-fold4-best.pt',
    'rdppimi-s4-3b-weighted-pair-fold5-best.pt',
    'proteinshake-ppi-esm2-35m-final.pt',
    'graphmvp-molecular-encoder-init.pt',
    'rdppimi-s4-3b-weighted-pair-results.md',
    'rdppimi-s4-3b-weighted-pair-results.json',
    'rdppimi-s4-3b-weighted-pair-grid-registry.csv',
    'rdppimi-s4-3b-weighted-pair-run-configs.tar.gz',
    'rdppimi-s4-3b-weighted-pair-minimal-assets.tar.gz',
    'SHA256SUMS.txt',
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256sums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        digest, filename = line.split(None, 1)
        checksums[filename.strip()] = digest
    return checksums


def verify_checksums(asset_dir: Path) -> None:
    checksum_path = asset_dir / 'SHA256SUMS.txt'
    checksums = parse_sha256sums(checksum_path)
    for filename, expected in checksums.items():
        path = asset_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f'Checksum entry is missing from asset dir: {filename}')
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f'SHA256 mismatch for {filename}: expected {expected}, got {actual}')


def verify_torch_load(asset_dir: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PyTorch is required to verify .pt files') from exc
    for filename in EXPECTED_FILES:
        if not filename.endswith('.pt'):
            continue
        path = asset_dir / filename
        obj = torch.load(path, map_location='cpu')
        if not hasattr(obj, 'keys'):
            raise RuntimeError(f'{filename} did not load as a mapping-like checkpoint')


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify RDPPIMI release assets.')
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--skip-torch-load', action='store_true')
    args = parser.parse_args()

    asset_dir = args.asset_dir.resolve()
    missing = [name for name in EXPECTED_FILES if not (asset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing release files: ' + ', '.join(missing))
    verify_checksums(asset_dir)
    if not args.skip_torch_load:
        verify_torch_load(asset_dir)
    print(f'Verified {len(EXPECTED_FILES)} release files in {asset_dir}')


if __name__ == '__main__':
    main()
