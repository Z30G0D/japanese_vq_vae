import numpy as np
import pytest

from jvqvae.data.splits import carve_validation, split_by_writer


def writer_ids(n_writers=10, n_classes=5):
    return np.repeat(np.arange(n_writers, dtype=np.int64), n_classes)


def test_splits_are_disjoint_and_cover_everything():
    ids = writer_ids()
    split = split_by_writer(ids, seed=0, n_val_writers=2, n_test_writers=2)

    all_indices = np.concatenate([split.train, split.val, split.test])
    assert sorted(all_indices.tolist()) == list(range(len(ids)))


def test_no_writer_appears_in_two_splits():
    ids = writer_ids()
    split = split_by_writer(ids, seed=0, n_val_writers=2, n_test_writers=2)

    train_w = set(ids[split.train].tolist())
    val_w = set(ids[split.val].tolist())
    test_w = set(ids[split.test].tolist())
    assert train_w & val_w == set()
    assert train_w & test_w == set()
    assert val_w & test_w == set()
    assert len(val_w) == 2 and len(test_w) == 2 and len(train_w) == 6


def test_split_is_deterministic_for_a_seed():
    ids = writer_ids()
    a = split_by_writer(ids, seed=3, n_val_writers=2, n_test_writers=2)
    b = split_by_writer(ids, seed=3, n_val_writers=2, n_test_writers=2)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)


def test_different_seeds_give_different_splits():
    ids = writer_ids()
    a = split_by_writer(ids, seed=1, n_val_writers=2, n_test_writers=2)
    b = split_by_writer(ids, seed=2, n_val_writers=2, n_test_writers=2)
    assert not np.array_equal(a.test, b.test)


def test_too_few_writers_is_rejected():
    ids = writer_ids(n_writers=3)
    with pytest.raises(ValueError, match="writers"):
        split_by_writer(ids, seed=0, n_val_writers=2, n_test_writers=2)


def test_every_split_contains_every_class():
    n_writers, n_classes = 10, 5
    ids = writer_ids(n_writers, n_classes)
    class_ids = np.tile(np.arange(n_classes, dtype=np.int64), n_writers)
    split = split_by_writer(ids, seed=0, n_val_writers=2, n_test_writers=2)

    for indices in (split.train, split.val, split.test):
        assert set(class_ids[indices].tolist()) == set(range(n_classes))


def test_carve_validation_is_disjoint_and_deterministic():
    train, val = carve_validation(100, seed=0, n_val=20)
    assert len(val) == 20 and len(train) == 80
    assert set(train.tolist()) & set(val.tolist()) == set()

    train2, val2 = carve_validation(100, seed=0, n_val=20)
    assert np.array_equal(val, val2)
