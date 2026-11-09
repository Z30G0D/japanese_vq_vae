import numpy as np
import pytest

from jvqvae.data.sources.k49 import K49_FILES, load_k49


@pytest.fixture
def fake_k49(tmp_path):
    rng = np.random.default_rng(0)
    # 3 classes x 4 samples, deliberately imbalanced to exercise max_per_class.
    labels = np.array([0, 0, 0, 0, 1, 1, 2], dtype=np.int64)
    images = rng.integers(0, 256, size=(len(labels), 28, 28), dtype=np.uint8)
    np.savez_compressed(tmp_path / "k49-train-imgs.npz", images)
    np.savez_compressed(tmp_path / "k49-train-labels.npz", labels)
    (tmp_path / "k49_classmap.csv").write_text(
        "index,codepoint,char\n0,U+3042,あ\n1,U+3044,い\n2,U+3046,う\n",
        encoding="utf-8",
    )
    return tmp_path


def test_urls_cover_the_files_we_need():
    assert "k49-train-imgs.npz" in K49_FILES
    assert "k49_classmap.csv" in K49_FILES
    assert all(url.startswith("https://") for url in K49_FILES.values())


def test_load_produces_padded_normalized_data(fake_k49):
    data = load_k49(fake_k49, split="train")

    assert len(data) == 7
    assert data.image_shape == (32, 32)
    assert data.images.dtype == np.float32
    assert data.images.min() >= 0.0 and data.images.max() <= 1.0
    assert data.writer_ids is None


def test_class_table_is_read_from_the_classmap(fake_k49):
    data = load_k49(fake_k49, split="train")
    assert data.classes == ["あ", "い", "う"]
    assert data.num_classes == 3


def test_padding_puts_original_pixels_in_the_middle(fake_k49):
    data = load_k49(fake_k49, split="train")
    # 28 -> 32 means a 2px border of zeros on every side.
    assert data.images[:, :2, :].sum() == 0.0
    assert data.images[:, -2:, :].sum() == 0.0
    assert data.images[:, :, :2].sum() == 0.0


def test_max_per_class_caps_each_class(fake_k49):
    data = load_k49(fake_k49, split="train", max_per_class=2)
    counts = np.bincount(data.class_ids, minlength=3)
    assert counts.tolist() == [2, 2, 1]


def test_subsampling_is_deterministic(fake_k49):
    a = load_k49(fake_k49, split="train", max_per_class=2, seed=5)
    b = load_k49(fake_k49, split="train", max_per_class=2, seed=5)
    assert np.array_equal(a.images, b.images)


def test_missing_files_give_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch-k49"):
        load_k49(tmp_path, split="train")
