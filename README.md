# Japanese handwriting with a VQ-VAE

A rebuild of a 2018 project that trained a convolutional VAE on ETL4 handwritten
hiragana. This version uses a VQ-VAE with a class-conditional transformer prior, so
you can ask for a specific character instead of sampling blindly.

Original write-up: [Computer made Japanese letters through Variational
Autoencoder](https://medium.com/@tomernahshon/computer-made-japanese-letters-through-variational-autoencoder-2fdb5b6a0990)

Design: `docs/superpowers/specs/2026-11-05-japanese-vq-vae-design.md`

## Setup

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

## Attribution

ETL4: Electrotechnical Laboratory, Japanese Technical Committee for Optical Character
Recognition, ETL Character Database, 1973-1984.

Kuzushiji-49: "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset"
(created by NIJL and others), doi:10.20676/00000341.

ETL record layout and character set details were cross-checked against the
MIT-licensed [choo/etlcdb-image-extractor](https://github.com/choo/etlcdb-image-extractor).
