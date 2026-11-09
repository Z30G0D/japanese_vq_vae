"""Command line entry points for data preparation, training, and sampling."""

import argparse
from dataclasses import replace
from pathlib import Path

from jvqvae.codes import extract_codes, load_codes, save_codes
from jvqvae.config import load_run_config
from jvqvae.data.pipeline import load_splits
from jvqvae.data.sources.etl4 import ingest_etl4
from jvqvae.data.sources.k49 import download_k49, load_k49
from jvqvae.device import select_device
from jvqvae.figures import (
    save_guidance_sweep,
    save_reconstruction_grid,
    save_sample_grid,
)
from jvqvae.models.prior import load_prior_checkpoint
from jvqvae.models.vqvae import load_checkpoint
from jvqvae.train.prior import train_prior
from jvqvae.train.vqvae import train_vqvae


def _fetch_k49(args: argparse.Namespace) -> int:
    directory = download_k49(args.dest)
    data = load_k49(directory, split="train", max_per_class=10)
    print(
        f"ready: {directory}, verified by loading {len(data)} images "
        f"across {data.num_classes} classes"
    )
    return 0


def _ingest_etl4(args: argparse.Namespace) -> int:
    config = load_run_config(args.config).data
    if config.etl4_zip is None:
        raise SystemExit(
            f"{args.config} does not set data.etl4_zip. Add a line such as "
            "'etl4_zip: data/raw/ETL4.zip' under the 'data:' section, pointing "
            "at the archive you downloaded from AIST."
        )

    cache = Path(config.cache_dir) / f"etl4_{config.image_size}.npz"
    try:
        data = ingest_etl4(
            config.etl4_zip,
            cache,
            image_size=config.image_size,
            verify=not args.no_verify,
            strict=not args.lenient,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"ingested {len(data)} images, {data.num_classes} classes, "
        f"{len(set(data.writer_ids.tolist()))} writers -> {cache}"
    )
    return 0


def _train_vqvae(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    if args.max_steps is not None:
        config = replace(config, train=replace(config.train, max_steps=args.max_steps))

    device = select_device(args.device or config.train.device)
    print(f"training {config.name} on {device}")

    splits = load_splits(config.data)
    record = train_vqvae(
        config, splits, device=device, output_root=args.output_root
    )

    results = Path(args.output_root) / config.train.results_dir / config.name
    model, _ = load_checkpoint(
        Path(args.output_root)
        / config.train.checkpoint_dir
        / config.name
        / "vqvae_best.pt",
        device=device,
    )
    figure = save_reconstruction_grid(
        model,
        splits.get("val") or splits["train"],
        results / "reconstructions.png",
        device=device,
    )

    print(
        f"done after {record['total_steps']} steps; "
        f"best val {record['best_val_loss']}; figure at {figure}"
    )
    return 0


def _extract_codes(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    device = select_device(args.device or config.train.device)

    checkpoint = Path(
        args.checkpoint
        or Path(args.output_root)
        / config.train.checkpoint_dir
        / config.name
        / "vqvae_best.pt"
    )
    if not checkpoint.exists():
        raise SystemExit(
            f"{checkpoint} does not exist. Run `jvqvae train-vqvae --config "
            f"{args.config}` first."
        )

    model, _ = load_checkpoint(checkpoint, device=device)
    splits = load_splits(config.data)
    if args.split not in splits:
        raise SystemExit(
            f"unknown split {args.split!r}; this config has: "
            f"{', '.join(sorted(splits))}"
        )

    codes = extract_codes(model, splits[args.split], device=device)
    out = Path(
        args.out
        or Path(args.output_root)
        / config.train.results_dir
        / config.name
        / f"codes_{args.split}.npz"
    )
    save_codes(codes, out)
    print(
        f"wrote {len(codes)} code grids of "
        f"{codes.latent_size}x{codes.latent_size} to {out}"
    )
    return 0


def _require_file(path: Path, command: str, config: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Run `jvqvae {command} --config {config}` first."
        )


def _train_prior(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    if args.max_steps is not None:
        config = replace(config, prior=replace(config.prior, max_steps=args.max_steps))

    device = select_device(args.device or config.train.device)
    codes_path = Path(
        args.codes
        or Path(args.output_root)
        / config.train.results_dir
        / config.name
        / "codes_train.npz"
    )
    _require_file(codes_path, "extract-codes", args.config)

    codes = load_codes(codes_path)
    print(f"training prior {config.name} on {device}")
    record = train_prior(
        config, {"train": codes}, device=device, output_root=args.output_root
    )
    print(
        f"done after {record['total_steps']} steps; "
        f"best val {record['best_val_loss']}"
    )
    return 0


def _sample(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    device = select_device(args.device or config.train.device)
    root = Path(args.output_root)
    name_dir = root / config.train.checkpoint_dir / config.name

    vqvae_path = Path(args.vqvae or name_dir / "vqvae_best.pt")
    prior_path = Path(args.prior or name_dir / "prior_best.pt")
    _require_file(vqvae_path, "train-vqvae", args.config)
    _require_file(prior_path, "train-prior", args.config)

    vqvae, _ = load_checkpoint(vqvae_path, device=device)
    prior, _ = load_prior_checkpoint(prior_path, device=device)
    class_ids = (
        args.class_ids
        if args.class_ids is not None
        else list(range(prior.num_classes))
    )

    results = root / config.train.results_dir / config.name
    figure = save_sample_grid(
        prior,
        vqvae,
        results / "samples.png",
        class_ids=class_ids,
        samples_per_class=args.n,
        device=device,
        temperature=args.temperature,
        top_p=args.top_p,
        guidance_scale=args.guidance,
        seed=args.seed,
    )
    written = [figure]
    if args.sweep_class is not None:
        written.append(
            save_guidance_sweep(
                prior,
                vqvae,
                results / "guidance_sweep.png",
                class_id=args.sweep_class,
                num_samples=args.n,
                device=device,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
        )
    print("wrote " + ", ".join(str(path) for path in written))
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

    train = subparsers.add_parser("train-vqvae", help="train the VQ-VAE")
    train.add_argument("--config", default="configs/etl4.yaml")
    train.add_argument("--device", default=None, help="cpu, mps, cuda or auto")
    train.add_argument("--max-steps", type=int, default=None, help="cap for a smoke run")
    train.add_argument("--output-root", default=".")
    train.set_defaults(handler=_train_vqvae)

    extract = subparsers.add_parser(
        "extract-codes", help="write the code grids the prior trains on"
    )
    extract.add_argument("--config", default="configs/etl4.yaml")
    extract.add_argument("--checkpoint", default=None)
    extract.add_argument("--split", default="train")
    extract.add_argument("--out", default=None)
    extract.add_argument("--device", default=None)
    extract.add_argument("--output-root", default=".")
    extract.set_defaults(handler=_extract_codes)

    prior = subparsers.add_parser(
        "train-prior", help="train the transformer prior on extracted codes"
    )
    prior.add_argument("--config", default="configs/etl4.yaml")
    prior.add_argument("--device", default=None, help="cpu, mps, cuda or auto")
    prior.add_argument("--max-steps", type=int, default=None, help="cap for a smoke run")
    prior.add_argument("--output-root", default=".")
    prior.add_argument(
        "--codes",
        default=None,
        help="code grids to train on (default: results/<name>/codes_train.npz)",
    )
    prior.set_defaults(handler=_train_prior)

    sample = subparsers.add_parser(
        "sample", help="decode prior samples with a trained VQ-VAE"
    )
    sample.add_argument("--config", default="configs/etl4.yaml")
    sample.add_argument("--device", default=None, help="cpu, mps, cuda or auto")
    sample.add_argument("--output-root", default=".")
    sample.add_argument("--vqvae", default=None, help="VQ-VAE checkpoint")
    sample.add_argument("--prior", default=None, help="prior checkpoint")
    sample.add_argument(
        "--class",
        dest="class_ids",
        action="append",
        type=int,
        default=None,
        help="class id to sample; repeat for several. Default: every class",
    )
    sample.add_argument("--n", type=int, default=4, help="samples per class")
    sample.add_argument("--guidance", type=float, default=None)
    sample.add_argument("--temperature", type=float, default=None)
    sample.add_argument("--top-p", type=float, default=None)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument(
        "--sweep-class",
        type=int,
        default=None,
        help="also write a guidance sweep for this class id",
    )
    sample.set_defaults(handler=_sample)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
