from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ppo_slm.data import load_target


def test_load_target_converts_rgb_resizes_and_normalizes(tmp_path: Path) -> None:
    pixels = np.zeros((16, 12, 3), dtype=np.uint8)
    pixels[3:12, 2:10] = (255, 128, 64)
    path = tmp_path / "rgb.png"
    Image.fromarray(pixels).save(path)

    target = load_target(path, image_size=32)

    assert target.shape == (1, 32, 32)
    assert target.dtype == torch.float32
    assert torch.isfinite(target).all()
    assert float(target.min()) >= 0.0
    assert float(target.max()) == pytest.approx(1.0)


def test_load_target_accepts_grayscale_image(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    Image.fromarray(np.array([[0, 64], [128, 255]], dtype=np.uint8)).save(path)

    target = load_target(path, image_size=2)

    assert target.shape == (1, 2, 2)
    assert target[0, 1, 1].item() == pytest.approx(1.0)


def test_load_target_rejects_missing_corrupt_and_black_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_target(tmp_path / "missing.bmp")

    corrupt = tmp_path / "corrupt.bmp"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="decode"):
        load_target(corrupt)

    black = tmp_path / "black.bmp"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(black)
    with pytest.raises(ValueError, match="black"):
        load_target(black)


def test_load_target_rejects_invalid_output_size(tmp_path: Path) -> None:
    path = tmp_path / "valid.bmp"
    Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(path)

    with pytest.raises(ValueError, match="image_size"):
        load_target(path, image_size=0)
