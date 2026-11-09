import numpy as np
import pytest

from jvqvae.data.sources.etl4_records import (
    HEIGHT,
    IMAGE_BYTES,
    RECORD_SIZE,
    WIDTH,
    decode_hiragana,
    encode_record,
    parse_record,
)


def test_layout_arithmetic_is_consistent():
    assert WIDTH * HEIGHT == IMAGE_BYTES * 2
    assert 9 + 1 + 206 + IMAGE_BYTES == RECORD_SIZE


def test_round_trip_recovers_code_and_pixels():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 16, size=(HEIGHT, WIDTH), dtype=np.uint8)

    code, decoded = parse_record(encode_record(0xB1, image))

    assert code == 0xB1
    assert decoded.shape == (HEIGHT, WIDTH)
    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, image)


def test_nibble_order_is_high_then_low():
    image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    image[0, 0] = 0xA
    image[0, 1] = 0x3

    _, decoded = parse_record(encode_record(0xB1, image))

    assert decoded[0, 0] == 0xA
    assert decoded[0, 1] == 0x3


def test_wrong_record_length_is_rejected():
    with pytest.raises(ValueError, match="2952"):
        parse_record(b"\x00" * 100)


@pytest.mark.parametrize(
    ("code", "expected"),
    [(0xB1, "あ"), (0xB2, "い"), (0xB3, "う"), (0xB4, "え"), (0xB5, "お")],
)
def test_known_codes_decode_to_hiragana(code, expected):
    assert decode_hiragana(code) == expected


def test_non_kana_codes_return_none():
    assert decode_hiragana(0x41) is None


def test_decoded_hiragana_are_distinct_across_the_kana_range():
    decoded = [decode_hiragana(code) for code in range(0xA1, 0xE0)]
    hiragana = [char for char in decoded if char is not None]
    assert len(hiragana) >= 45
    assert len(set(hiragana)) == len(hiragana)
