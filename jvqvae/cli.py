"""Command line entry points for data preparation."""

import argparse

from jvqvae.config import load_data_config
from jvqvae.data.sources.etl4 import ingest_etl4
from jvqvae.data.sources.k49 import download_k49, load_k49


def _fetch_k49(args: argparse.Namespace) -> int:
    directory = download_k49(args.dest)
    data = load_k49(directory, split="train", max_per_class=10)
    print(
        f"ready: {directory}, verified by loading {len(data)} images "
        f"across {data.num_classes} classes"
    )
    return 0


def _ingest_etl4(args: argparse.Namespace) -> int:
    config = load_data_config(args.config)
    if config.etl4_zip is None:
        raise SystemExit(
            f"{args.config} does not set etl4_zip. Add a line such as "
            "'etl4_zip: data/raw/ETL4.zip' pointing at the archive you "
            "downloaded from AIST."
        )

    cache = f"{config.cache_dir}/etl4_{config.image_size}.npz"
    data = ingest_etl4(
        config.etl4_zip,
        cache,
        image_size=config.image_size,
        verify=not args.no_verify,
        strict=not args.lenient,
    )
    print(
        f"ingested {len(data)} images, {data.num_classes} classes, "
        f"{len(set(data.writer_ids.tolist()))} writers -> {cache}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jvqvae")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch-k49", help="download Kuzushiji-49")
    fetch.add_argument("--dest", default="data/k49")
    fetch.set_defaults(handler=_fetch_k49)

    ingest = subparsers.add_parser(
        "ingest-etl4", help="verify and ingest a user-supplied ETL4 zip"
    )
    ingest.add_argument("--config", default="configs/etl4.yaml")
    ingest.add_argument(
        "--no-verify", action="store_true", help="skip the size and MD5 check"
    )
    ingest.add_argument(
        "--lenient", action="store_true", help="skip record and class count assertions"
    )
    ingest.set_defaults(handler=_ingest_etl4)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
