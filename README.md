# Japanese handwriting with a VQ-VAE

A rebuild of a 2018 project that trained a convolutional VAE on ETL4 handwritten
hiragana. This version uses a VQ-VAE with a class-conditional transformer prior, so
you can ask for a specific character instead of sampling blindly.

**Status: the data pipeline is complete; the VQ-VAE and the prior are not written
yet.** Everything below works today. There is no training or sampling command.

Original write-up: [Computer made Japanese letters through Variational
Autoencoder](https://medium.com/@tomernahshon/computer-made-japanese-letters-through-variational-autoencoder-2fdb5b6a0990)

Design: `docs/superpowers/specs/2026-11-05-japanese-vq-vae-design.md`

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --python 3.12
uv run pytest
```

## Data

### Kuzushiji-49 (development)

Downloads without registration:

```bash
uv run jvqvae fetch-k49 --dest data/k49
```

### ETL4 (the real dataset)

ETL4 cannot be downloaded by this repository. AIST's Terms of Use permit
distribution only through their own site, so you must obtain it yourself:

1. Go to <https://etlcdb.db.aist.go.jp/>, accept the Terms of Use, and register to
   receive the download password.
2. Download ETL4 and place the zip at `data/raw/ETL4.zip`.
3. Run `uv run jvqvae ingest-etl4 --config configs/etl4.yaml`

Ingestion verifies the archive against AIST's published size and MD5 before parsing,
and asserts that it contains 6,120 records across 51 classes with 120 samples each.
If your archive differs, `--no-verify` skips the size and MD5 check and `--lenient`
skips the count assertions. Reach for them only to inspect a file that fails, not as
a habit: they exist to make a mismatch debuggable, not to paper over one.

## Using the data

`load_splits` takes a config and returns train, val and test; `CharacterTorchDataset`
wraps any of them for a `DataLoader`, yielding a `(1, H, W)` float tensor and an
integer class id:

```python
from torch.utils.data import DataLoader

from jvqvae.config import load_data_config
from jvqvae.data.pipeline import load_splits
from jvqvae.data.torch_dataset import CharacterTorchDataset

splits = load_splits(load_data_config("configs/etl4.yaml"))
loader = DataLoader(CharacterTorchDataset(splits["train"]), batch_size=32, shuffle=True)
```

How a source gets divided is decided in one place, by the source itself. ETL4 carries
writer identities, so it is always split by writer: 96 writers train, 12 validate, 12
test, and no hand appears in two splits. K49 has no writer identities and ships its
own test set, so its validation set is carved at random out of the official train
portion. Asking for a writer split on a source without writers is an error, not a
silently random split.

## Attribution

ETL4: Electrotechnical Laboratory, Japanese Technical Committee for Optical Character
Recognition, ETL Character Database, 1973-1984. Redistribution is not permitted, so
no ETL data appears in this repository.

Kuzushiji-49: "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset"
(created by NIJL and others), doi:10.20676/00000341. Licensed CC BY-SA 4.0.

ETL record layout and character set details were cross-checked against the
MIT-licensed [choo/etlcdb-image-extractor](https://github.com/choo/etlcdb-image-extractor).

## Licence

The code in this repository is MIT licensed; see `LICENSE`. The datasets are not
covered by it and carry their own terms, noted above.
