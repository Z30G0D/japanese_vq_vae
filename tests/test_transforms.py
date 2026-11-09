import numpy as np

from jvqvae.data.transforms import pad_to_square, resize


def test_pad_to_square_centers_content():
    images = np.ones((2, 2, 4), dtype=np.float32)
    padded = pad_to_square(images)
    assert padded.shape == (2, 4, 4)
    # One row of zeros added above and below the original two rows.
    assert padded[0, 0].sum() == 0.0
    assert padded[0, 3].sum() == 0.0
    assert padded[0, 1].sum() == 4.0


def test_pad_to_square_is_a_noop_when_already_square():
    images = np.ones((1, 3, 3), dtype=np.float32)
    assert pad_to_square(images).shape == (1, 3, 3)


def test_resize_changes_shape_and_preserves_range():
    rng = np.random.default_rng(0)
    images = rng.random((3, 16, 16)).astype(np.float32)
    out = resize(images, 8)
    assert out.shape == (3, 8, 8)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resize_preserves_a_constant_image():
    images = np.full((1, 16, 16), 0.5, dtype=np.float32)
    out = resize(images, 8)
    assert np.allclose(out, 0.5, atol=1e-6)
