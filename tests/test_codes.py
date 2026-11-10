import numpy as np
import pytest
import torch

from jvqvae.codes import CodeData, extract_codes, load_codes, save_codes
from jvqvae.models.vqvae import VQVAE

CPU = torch.device("cpu")


def test_extraction_produces_one_grid_per_image(tiny_config, tiny_data):
    model = VQVAE(tiny_config.model)

    codes = extract_codes(model, tiny_data, device=CPU, batch_size=5)

    assert codes.codes.shape == (12, 4, 4)
    assert codes.codes.dtype == np.int64
    assert codes.latent_size == 4


def test_extracted_codes_stay_inside_the_codebook(tiny_config, tiny_data):
    model = VQVAE(tiny_config.model)

    codes = extract_codes(model, tiny_data, device=CPU, batch_size=5)

    assert codes.codes.min() >= 0
    assert codes.codes.max() < tiny_config.model.num_codes
    assert codes.num_codes == tiny_config.model.num_codes


def test_labels_survive_extraction_in_order(tiny_config, tiny_data):
    model = VQVAE(tiny_config.model)

    codes = extract_codes(model, tiny_data, device=CPU, batch_size=5)

    assert np.array_equal(codes.class_ids, tiny_data.class_ids)
    assert codes.classes == tiny_data.classes


def test_batch_size_does_not_change_the_result(tiny_config, tiny_data):
    """A ragged final batch is where an off-by-one would show up."""
    model = VQVAE(tiny_config.model).eval()

    one = extract_codes(model, tiny_data, device=CPU, batch_size=12)
    many = extract_codes(model, tiny_data, device=CPU, batch_size=5)

    assert np.array_equal(one.codes, many.codes)


def test_extraction_leaves_the_codebook_untouched(tiny_config, tiny_data):
    """Extraction must not keep training the quantizer."""
    model = VQVAE(tiny_config.model).train()
    before = model.quantizer.codebook.clone()

    extract_codes(model, tiny_data, device=CPU)

    assert torch.equal(model.quantizer.codebook, before)


def test_codes_round_trip_through_a_file(tiny_config, tiny_data, tmp_path):
    model = VQVAE(tiny_config.model)
    original = extract_codes(model, tiny_data, device=CPU)

    path = tmp_path / "codes.npz"
    save_codes(original, path)
    restored = load_codes(path)

    assert np.array_equal(restored.codes, original.codes)
    assert np.array_equal(restored.class_ids, original.class_ids)
    assert restored.classes == original.classes
    assert restored.num_codes == original.num_codes
    assert restored.latent_size == original.latent_size


def test_the_saved_file_does_not_need_pickle(tiny_config, tiny_data, tmp_path):
    model = VQVAE(tiny_config.model)
    path = tmp_path / "codes.npz"
    save_codes(extract_codes(model, tiny_data, device=CPU), path)

    with np.load(path, allow_pickle=False) as raw:
        assert "codes" in raw


def test_code_data_rejects_a_mismatched_length():
    with pytest.raises(ValueError, match="length"):
        CodeData(
            codes=np.zeros((3, 4, 4), dtype=np.int64),
            class_ids=np.zeros(2, dtype=np.int64),
            classes=["あ"],
            num_codes=8,
        )


def test_code_data_rejects_codes_outside_the_codebook():
    with pytest.raises(ValueError, match="codebook"):
        CodeData(
            codes=np.full((2, 4, 4), 99, dtype=np.int64),
            class_ids=np.zeros(2, dtype=np.int64),
            classes=["あ"],
            num_codes=8,
        )


def test_extraction_from_an_empty_dataset_keeps_the_grid_size(tiny_config, tiny_data):
    empty = tiny_data.subset(np.arange(0))
    codes = extract_codes(VQVAE(tiny_config.model), empty, device=CPU)

    assert codes.codes.shape == (0, 4, 4)
    assert codes.latent_size == 4


def test_code_data_rejects_a_non_square_grid():
    with pytest.raises(ValueError, match="square"):
        CodeData(
            codes=np.zeros((2, 4, 5), dtype=np.int64),
            class_ids=np.zeros(2, dtype=np.int64),
            classes=["あ"],
            num_codes=8,
        )
