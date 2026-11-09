import numpy as np
import pytest

from jvqvae.data.dataset import CharacterData


def make(n=6, n_classes=3):
    return CharacterData(
        images=np.zeros((n, 4, 4), dtype=np.float32),
        class_ids=np.arange(n, dtype=np.int64) % n_classes,
        writer_ids=np.arange(n, dtype=np.int64) // n_classes,
        classes=[chr(0x3042 + i) for i in range(n_classes)],
    )


def test_basic_properties():
    data = make()
    assert len(data) == 6
    assert data.num_classes == 3
    assert data.image_shape == (4, 4)


def test_subset_keeps_alignment():
    data = make()
    subset = data.subset(np.array([0, 4]))
    assert len(subset) == 2
    assert subset.class_ids.tolist() == [0, 1]
    assert subset.writer_ids.tolist() == [0, 1]
    assert subset.classes == data.classes


def test_writer_ids_may_be_absent():
    data = CharacterData(
        images=np.zeros((2, 4, 4), dtype=np.float32),
        class_ids=np.zeros(2, dtype=np.int64),
        writer_ids=None,
        classes=["あ"],
    )
    assert data.writer_ids is None
    assert data.subset(np.array([1])).writer_ids is None


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="length"):
        CharacterData(
            images=np.zeros((3, 4, 4), dtype=np.float32),
            class_ids=np.zeros(2, dtype=np.int64),
            writer_ids=None,
            classes=["あ"],
        )


def test_out_of_range_pixels_are_rejected():
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        CharacterData(
            images=np.full((2, 4, 4), 255.0, dtype=np.float32),
            class_ids=np.zeros(2, dtype=np.int64),
            writer_ids=None,
            classes=["あ"],
        )


def test_class_id_beyond_class_table_is_rejected():
    with pytest.raises(ValueError, match="class id"):
        CharacterData(
            images=np.zeros((2, 4, 4), dtype=np.float32),
            class_ids=np.array([0, 9], dtype=np.int64),
            writer_ids=None,
            classes=["あ"],
        )
