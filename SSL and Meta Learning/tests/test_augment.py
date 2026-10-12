import torch

from ssl_methods.augment import (DEFAULT_AUG, MILD_AUG, GPUAugment, color_jitter, gaussian_blur, hsv_to_rgb,
                                 normalize_uint8, random_grayscale, random_resized_crop, rgb_to_hsv, solarize)


def _batch(b=6, c=3, s=32):
    torch.manual_seed(0)
    return torch.randint(0, 256, (b, c, s, s), dtype=torch.uint8)


def test_full_pipeline_shapes_dtype_range():
    x = _batch()
    aug = GPUAugment({**DEFAULT_AUG, "solarize_p": 0.5, "blur_p": 1.0}, [0.0] * 3, [1.0] * 3, 32)
    y = aug(x)
    assert y.shape == (6, 3, 32, 32) and y.dtype == torch.float32
    assert torch.isfinite(y).all()
    assert y.min() >= -1e-5 and y.max() <= 1 + 1e-5      # mean 0 / std 1 => output stays in [0,1]


def test_other_output_size_and_mild():
    x = _batch(s=40)
    y = GPUAugment(MILD_AUG, [0.5] * 3, [0.25] * 3, 40)(x, out_size=16)
    assert y.shape == (6, 3, 16, 16)


def test_ops_keep_range():
    x = _batch().float() / 255
    for f in (lambda t: random_resized_crop(t, 24, (0.08, 1), (0.75, 1.33), 0.5),
              lambda t: color_jitter(t, 1.0, 0.4, 0.4, 0.4, 0.1),
              lambda t: random_grayscale(t, 1.0),
              lambda t: gaussian_blur(t, 1.0, (0.1, 2.0)),
              lambda t: solarize(t, 1.0)):
        y = f(x)
        assert y.min() >= -1e-5 and y.max() <= 1 + 1e-5 and y.dtype == torch.float32


def test_full_crop_identity():
    """scale=1, ratio=1, no flip must reproduce the image (checks the affine-grid convention)."""
    x = _batch().float() / 255
    y = random_resized_crop(x, 32, (1.0, 1.0), (1.0, 1.0), 0.0)
    assert torch.allclose(x, y, atol=1e-4)


def test_flip_only():
    x = _batch().float() / 255
    y = random_resized_crop(x, 32, (1.0, 1.0), (1.0, 1.0), 1.0)
    assert torch.allclose(torch.flip(x, dims=[3]), y, atol=1e-4)


def test_hsv_roundtrip():
    x = torch.rand(4, 3, 8, 8)
    assert torch.allclose(hsv_to_rgb(rgb_to_hsv(x)), x, atol=1e-4)


def test_grayscale_makes_equal_channels():
    y = random_grayscale(torch.rand(3, 3, 8, 8), 1.0)
    assert torch.allclose(y[:, 0], y[:, 1]) and torch.allclose(y[:, 1], y[:, 2])


def test_single_channel_input():
    x = torch.randint(0, 256, (4, 1, 16, 16), dtype=torch.uint8)
    y = GPUAugment(DEFAULT_AUG, [0.5], [0.25], 16)(x)
    assert y.shape == (4, 1, 16, 16) and torch.isfinite(y).all()


def test_normalize_uint8():
    x = torch.full((1, 3, 2, 2), 255, dtype=torch.uint8)
    y = normalize_uint8(x, [0.5] * 3, [0.5] * 3)
    assert torch.allclose(y, torch.ones_like(y))
