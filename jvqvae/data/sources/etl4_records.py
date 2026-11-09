"""ETL4 C-type record parsing.

Record layout (struct ">9xB206x2736s"): 9 padding bytes, one character-code
byte, 206 padding bytes, then a 72x76 image at 4 bits per pixel.

The layout and the JIS X 0201 character set are documented by AIST and were
cross-checked against the MIT-licensed choo/etlcdb-image-extractor
(https://github.com/choo/etlcdb-image-extractor).
"""

import unicodedata

import numpy as np

RECORD_SIZE = 2952
IMAGE_BYTES = 2736
WIDTH = 72
HEIGHT = 76
CHAR_CODE_OFFSET = 9
MAX_PIXEL_VALUE = 15

_IMAGE_OFFSET = RECORD_SIZE - IMAGE_BYTES

# JIS X 0201 halfwidth katakana occupy 0xA1-0xDF, mapping to U+FF61-U+FF9F.
_HALFWIDTH_KATAKANA_START = 0xA1
_HALFWIDTH_KATAKANA_END = 0xDF
_HALFWIDTH_UNICODE_START = 0xFF61
# Distance from a katakana codepoint to its hiragana counterpart.
_KATAKANA_TO_HIRAGANA = 0x60


def parse_record(raw: bytes) -> tuple[int, np.ndarray]:
    """Return ``(character_code, image)`` for one 2,952-byte record."""
    if len(raw) != RECORD_SIZE:
        raise ValueError(f"record must be {RECORD_SIZE} bytes, got {len(raw)}")

    char_code = raw[CHAR_CODE_OFFSET]
    packed = np.frombuffer(raw[_IMAGE_OFFSET:], dtype=np.uint8)

    pixels = np.empty(packed.size * 2, dtype=np.uint8)
    pixels[0::2] = packed >> 4
    pixels[1::2] = packed & 0x0F
    return char_code, pixels.reshape(HEIGHT, WIDTH)


def encode_record(char_code: int, image: np.ndarray) -> bytes:
    """Build a synthetic record. Used by tests while AIST is unreachable."""
    if image.shape != (HEIGHT, WIDTH):
        raise ValueError(f"image must be ({HEIGHT}, {WIDTH}), got {image.shape}")
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"image must have an integer dtype, got {image.dtype}")
    if image.min(initial=0) < 0 or image.max(initial=0) > MAX_PIXEL_VALUE:
        raise ValueError("pixel values must fit in 4 bits")
    if not 0 <= char_code <= 0xFF:
        raise ValueError(f"char_code must be a single byte, got {char_code}")

    flat = image.reshape(-1).astype(np.uint8)
    packed = ((flat[0::2] << 4) | flat[1::2]).astype(np.uint8)
    padding = _IMAGE_OFFSET - CHAR_CODE_OFFSET - 1
    return (
        b"\x00" * CHAR_CODE_OFFSET
        + bytes([char_code])
        + b"\x00" * padding
        + packed.tobytes()
    )


def decode_hiragana(char_code: int) -> str | None:
    """Map a JIS X 0201 character code to hiragana, or ``None`` if not kana.

    ETL4 holds hiragana but encodes them with JIS X 0201, which only defines
    halfwidth katakana. NFKC widens the halfwidth form, and subtracting 0x60
    converts katakana to hiragana.
    """
    if not _HALFWIDTH_KATAKANA_START <= char_code <= _HALFWIDTH_KATAKANA_END:
        return None

    halfwidth = chr(_HALFWIDTH_UNICODE_START + char_code - _HALFWIDTH_KATAKANA_START)
    widened = unicodedata.normalize("NFKC", halfwidth)
    if len(widened) != 1:
        return None

    codepoint = ord(widened)
    if not 0x30A1 <= codepoint <= 0x30F6:
        return None
    return chr(codepoint - _KATAKANA_TO_HIRAGANA)
