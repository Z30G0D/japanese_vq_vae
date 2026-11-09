"""ETL4 ingestion.

The repository never ships or downloads ETL4. AIST's Terms of Use clause 5
prohibits redistribution and direct URL links, so the user registers at
https://etlcdb.db.aist.go.jp/ and supplies the zip. Ingestion verifies it
against AIST's published fixity before parsing a byte.

Citation required by clause 4: Electrotechnical Laboratory, Japanese Technical
Committee for Optical Character Recognition, ETL Character Database, 1973-1984.
"""

import hashlib
import zipfile
from pathlib import Path

import numpy as np

from jvqvae.data.dataset import CharacterData
from jvqvae.data.sources.etl4_records import (
    MAX_PIXEL_VALUE,
    RECORD_SIZE,
    decode_hiragana,
    parse_record,
)
from jvqvae.data.transforms import pad_to_square, resize

EXPECTED_ZIP_BYTES = 4_949_352
EXPECTED_ZIP_MD5 = "cede897313b3617e364ddb17b8fad499"
EXPECTED_RECORDS = 6120
EXPECTED_CLASSES = 51
EXPECTED_WRITERS = 120

_ACQUISITION_HINT = (
    "Register at https://etlcdb.db.aist.go.jp/ to obtain ETL4. The repository "
    "cannot download or redistribute it."
)


def verify_zip(path: str | Path) -> None:
    """Check the supplied archive against AIST's published size and MD5."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. {_ACQUISITION_HINT}")

    actual_size = path.stat().st_size
    if actual_size != EXPECTED_ZIP_BYTES:
        raise ValueError(
            f"unexpected size for {path}: got {actual_size} bytes, "
            f"expected {EXPECTED_ZIP_BYTES}. {_ACQUISITION_HINT}"
        )

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_ZIP_MD5:
        raise ValueError(
            f"MD5 mismatch for {path}: got {digest.hexdigest()}, "
            f"expected {EXPECTED_ZIP_MD5}. {_ACQUISITION_HINT}"
        )


def parse_etl4_stream(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse concatenated fixed-size records into images and character codes."""
    if len(raw) % RECORD_SIZE != 0:
        raise ValueError(
            f"data length {len(raw)} is not a multiple of the "
            f"{RECORD_SIZE}-byte record size"
        )

    count = len(raw) // RECORD_SIZE
    images, codes = [], []
    for index in range(count):
        code, image = parse_record(raw[index * RECORD_SIZE : (index + 1) * RECORD_SIZE])
        codes.append(code)
        images.append(image)
    return np.stack(images), np.array(codes, dtype=np.int64)


def _groups_are_complete(
    class_ids: np.ndarray, group_ids: np.ndarray, n_classes: int
) -> bool:
    for group in np.unique(group_ids):
        members = class_ids[group_ids == group]
        if len(np.unique(members)) != n_classes or len(members) != n_classes:
            return False
    return True


def infer_writer_ids(
    class_ids: np.ndarray, n_classes: int, n_writers: int
) -> np.ndarray:
    """Recover writer identity from record order, validating the assumption.

    ETL4 is documented as writer-major, but rather than trusting that, both
    plausible orderings are tested against the decoded class sequence and the
    one that produces complete per-writer groups wins.
    """
    count = len(class_ids)
    candidates = {
        "writer-major": np.arange(count) // n_classes,
        "character-major": np.arange(count) % n_writers,
    }
    for group_ids in candidates.values():
        if _groups_are_complete(class_ids, group_ids, n_classes):
            return group_ids.astype(np.int64)

    raise ValueError(
        "could not determine writer order: neither writer-major nor "
        "character-major grouping yields one sample per class per writer"
    )


def _check_strict(class_ids: np.ndarray, classes: list[str]) -> None:
    if len(class_ids) != EXPECTED_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_RECORDS} records, found {len(class_ids)}. "
            "The 2018 notebook read 6113 and silently dropped seven."
        )
    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(f"expected {EXPECTED_CLASSES} classes, found {len(classes)}")

    counts = np.bincount(class_ids, minlength=len(classes))
    if not np.all(counts == EXPECTED_WRITERS):
        offenders = {
            classes[i]: int(count)
            for i, count in enumerate(counts)
            if count != EXPECTED_WRITERS
        }
        raise ValueError(
            f"expected {EXPECTED_WRITERS} samples per class; these differ: {offenders}"
        )


def ingest_etl4(
    zip_path: str | Path,
    cache_path: str | Path,
    *,
    image_size: int = 64,
    verify: bool = True,
    strict: bool = True,
) -> CharacterData:
    """Verify, parse and cache ETL4 into a `CharacterData`."""
    if verify:
        verify_zip(zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        data_members = [
            name for name in members if "INFO" not in Path(name).name.upper()
        ]
        if len(data_members) != 1:
            raise ValueError(
                f"expected exactly one data member in {zip_path}, "
                f"found {data_members}"
            )
        raw = archive.read(data_members[0])

    raw_images, char_codes = parse_etl4_stream(raw)

    unique_codes = np.unique(char_codes)
    class_ids = np.searchsorted(unique_codes, char_codes).astype(np.int64)
    classes = [decode_hiragana(int(code)) or f"U+{code:04X}" for code in unique_codes]

    if strict:
        _check_strict(class_ids, classes)

    n_writers = len(class_ids) // max(len(classes), 1)
    writer_ids = infer_writer_ids(class_ids, len(classes), n_writers)

    images = raw_images.astype(np.float32) / float(MAX_PIXEL_VALUE)
    images = resize(pad_to_square(images), image_size)

    data = CharacterData(
        images=images,
        class_ids=class_ids,
        writer_ids=writer_ids,
        classes=classes,
    )
    _save_cache(data, cache_path)
    return data


def _cache_path(cache_path: str | Path) -> Path:
    """Apply the `.npz` suffix numpy would otherwise append behind our back."""
    path = Path(cache_path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    return path


def _save_cache(data: CharacterData, cache_path: str | Path) -> None:
    path = _cache_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "images": data.images,
        "class_ids": data.class_ids,
        "classes": np.array(data.classes),
    }
    if data.writer_ids is not None:
        arrays["writer_ids"] = data.writer_ids
    np.savez_compressed(path, **arrays)


def load_cached_etl4(cache_path: str | Path) -> CharacterData:
    with np.load(_cache_path(cache_path)) as cached:
        writer_ids = cached["writer_ids"] if "writer_ids" in cached else None
        return CharacterData(
            images=cached["images"],
            class_ids=cached["class_ids"],
            writer_ids=writer_ids,
            classes=[str(char) for char in cached["classes"]],
        )
