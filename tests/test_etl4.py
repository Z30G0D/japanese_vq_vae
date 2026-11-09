import hashlib
import zipfile

import numpy as np
import pytest

from jvqvae.data.dataset import CharacterData
from jvqvae.data.sources.etl4 import (
    EXPECTED_ZIP_BYTES,
    EXPECTED_ZIP_MD5,
    _save_cache,
    infer_writer_ids,
    ingest_etl4,
    load_cached_etl4,
    parse_etl4_stream,
    verify_zip,
)
from jvqvae.data.sources.etl4_records import HEIGHT, WIDTH, encode_record

N_CLASSES = 4
N_WRITERS = 3
CODES = [0xB1, 0xB2, 0xB3, 0xB4]  # あいうえ


def build_stream(writer_major=True):
    rng = np.random.default_rng(0)
    records = []
    if writer_major:
        order = [(w, c) for w in range(N_WRITERS) for c in range(N_CLASSES)]
    else:
        order = [(w, c) for c in range(N_CLASSES) for w in range(N_WRITERS)]
    for _, class_index in order:
        image = rng.integers(0, 16, size=(HEIGHT, WIDTH), dtype=np.uint8)
        records.append(encode_record(CODES[class_index], image))
    return b"".join(records)


@pytest.fixture
def fake_zip(tmp_path):
    path = tmp_path / "ETL4.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ETL4/ETL4C", build_stream())
    return path


def test_expected_fixity_constants_match_aist_listing():
    assert EXPECTED_ZIP_BYTES == 4_949_352
    assert EXPECTED_ZIP_MD5 == "cede897313b3617e364ddb17b8fad499"


def test_verify_rejects_a_wrong_sized_file(tmp_path):
    path = tmp_path / "ETL4.zip"
    path.write_bytes(b"nope")
    with pytest.raises(ValueError, match="size"):
        verify_zip(path)


def test_verify_accepts_a_file_matching_size_and_md5(tmp_path, monkeypatch):
    payload = b"x" * 32
    path = tmp_path / "ETL4.zip"
    path.write_bytes(payload)
    monkeypatch.setattr(
        "jvqvae.data.sources.etl4.EXPECTED_ZIP_BYTES", len(payload)
    )
    monkeypatch.setattr(
        "jvqvae.data.sources.etl4.EXPECTED_ZIP_MD5",
        hashlib.md5(payload).hexdigest(),
    )
    verify_zip(path)  # must not raise


def test_parse_stream_returns_all_records():
    images, codes = parse_etl4_stream(build_stream())
    assert images.shape == (N_CLASSES * N_WRITERS, HEIGHT, WIDTH)
    assert sorted(set(codes.tolist())) == CODES


def test_truncated_stream_is_rejected():
    with pytest.raises(ValueError, match="multiple"):
        parse_etl4_stream(build_stream()[:-10])


def test_writer_major_order_is_detected():
    class_ids = np.array([0, 1, 2, 3] * N_WRITERS, dtype=np.int64)
    writers = infer_writer_ids(class_ids, N_CLASSES, N_WRITERS)
    assert writers.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_character_major_order_is_detected():
    class_ids = np.repeat(np.arange(N_CLASSES), N_WRITERS).astype(np.int64)
    writers = infer_writer_ids(class_ids, N_CLASSES, N_WRITERS)
    assert writers.tolist() == [0, 1, 2] * N_CLASSES


def test_unrecognisable_order_raises():
    class_ids = np.zeros(N_CLASSES * N_WRITERS, dtype=np.int64)
    with pytest.raises(ValueError, match="writer order"):
        infer_writer_ids(class_ids, N_CLASSES, N_WRITERS)


def test_ingest_produces_normalized_square_images(fake_zip, tmp_path):
    data = ingest_etl4(
        fake_zip, tmp_path / "etl4.npz", image_size=64, verify=False, strict=False
    )

    assert len(data) == N_CLASSES * N_WRITERS
    assert data.image_shape == (64, 64)
    assert data.images.dtype == np.float32
    assert data.images.min() >= 0.0 and data.images.max() <= 1.0
    assert data.num_classes == N_CLASSES
    assert data.classes == ["あ", "い", "う", "え"]
    assert data.writer_ids is not None
    assert sorted(set(data.writer_ids.tolist())) == [0, 1, 2]


def test_ingest_writes_a_reloadable_cache(fake_zip, tmp_path):
    cache = tmp_path / "etl4.npz"
    original = ingest_etl4(fake_zip, cache, verify=False, strict=False)

    assert cache.exists()
    reloaded = load_cached_etl4(cache)
    assert np.array_equal(original.images, reloaded.images)
    assert np.array_equal(original.class_ids, reloaded.class_ids)
    assert np.array_equal(original.writer_ids, reloaded.writer_ids)
    assert original.classes == reloaded.classes


def test_cache_round_trip_adds_the_npz_suffix(fake_zip, tmp_path):
    original = ingest_etl4(fake_zip, tmp_path / "etl4", verify=False, strict=False)

    assert (tmp_path / "etl4.npz").exists()
    reloaded = load_cached_etl4(tmp_path / "etl4")
    assert np.array_equal(original.images, reloaded.images)
    assert original.classes == reloaded.classes


def test_cache_round_trip_preserves_absent_writer_ids(tmp_path):
    data = CharacterData(
        images=np.zeros((2, 4, 4), dtype=np.float32),
        class_ids=np.array([0, 1], dtype=np.int64),
        writer_ids=None,
        classes=["あ", "い"],
    )
    _save_cache(data, tmp_path / "nowriters.npz")

    assert load_cached_etl4(tmp_path / "nowriters.npz").writer_ids is None


def test_the_info_member_is_not_mistaken_for_data(tmp_path):
    path = tmp_path / "ETL4.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ETL4/ETL4C", build_stream())
        archive.writestr("ETL4/ETL4INFO", "writer metadata, not records")

    data = ingest_etl4(path, tmp_path / "etl4.npz", verify=False, strict=False)
    assert len(data) == N_CLASSES * N_WRITERS


def test_an_ambiguous_archive_is_rejected(tmp_path):
    path = tmp_path / "ETL4.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ETL4/ETL4C", build_stream())
        archive.writestr("ETL4/ETL4C_copy", build_stream())

    with pytest.raises(ValueError, match="exactly one data member"):
        ingest_etl4(path, tmp_path / "etl4.npz", verify=False, strict=False)


def test_strict_mode_rejects_a_short_archive(fake_zip, tmp_path):
    with pytest.raises(ValueError, match="6120"):
        ingest_etl4(fake_zip, tmp_path / "etl4.npz", verify=False, strict=True)
