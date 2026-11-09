import pytest

from jvqvae.cli import main


def test_no_subcommand_exits_with_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "fetch-k49" in out
    assert "train-prior" in out
    assert "sample" in out


def test_ingest_reports_the_configured_defaults(monkeypatch, tmp_path):
    captured = {}

    def fake_ingest(zip_path, cache, **kwargs):
        captured["zip_path"] = zip_path
        captured["cache"] = cache
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr("jvqvae.cli.ingest_etl4", fake_ingest)
    config = tmp_path / "etl4.yaml"
    config.write_text(
        "data:\n  source: etl4\n  image_size: 64\n"
        f"  cache_dir: {tmp_path}\n  etl4_zip: {tmp_path}/ETL4.zip\n"
    )

    with pytest.raises(SystemExit):
        main(["ingest-etl4", "--config", str(config)])

    from pathlib import Path

    assert captured["cache"] == Path(tmp_path) / "etl4_64.npz"
    assert captured["verify"] is True
    assert captured["strict"] is True


def test_no_verify_and_lenient_turn_the_checks_off(monkeypatch, tmp_path):
    captured = {}

    def fake_ingest(zip_path, cache, **kwargs):
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr("jvqvae.cli.ingest_etl4", fake_ingest)
    config = tmp_path / "etl4.yaml"
    config.write_text(
        "data:\n  source: etl4\n  image_size: 64\n"
        f"  cache_dir: {tmp_path}\n  etl4_zip: {tmp_path}/ETL4.zip\n"
    )

    with pytest.raises(SystemExit):
        main(["ingest-etl4", "--config", str(config), "--no-verify", "--lenient"])

    assert captured["verify"] is False
    assert captured["strict"] is False


def test_a_config_without_a_zip_path_says_what_to_add(tmp_path):
    config = tmp_path / "etl4.yaml"
    config.write_text("data:\n  source: etl4\n  image_size: 64\n")

    with pytest.raises(SystemExit, match="etl4_zip: data/raw/ETL4.zip"):
        main(["ingest-etl4", "--config", str(config)])


def test_missing_etl4_zip_exits_cleanly(monkeypatch, tmp_path):
    config = tmp_path / "etl4.yaml"
    config.write_text(
        "data:\n  source: etl4\n  image_size: 64\n"
        f"  cache_dir: {tmp_path}\n  etl4_zip: data/raw/ETL4.zip\n"
    )

    def fake_ingest(*args, **kwargs):
        raise FileNotFoundError(
            "data/raw/ETL4.zip does not exist. Register at https://etlcdb.db.aist.go.jp/"
        )

    monkeypatch.setattr("jvqvae.cli.ingest_etl4", fake_ingest)

    with pytest.raises(SystemExit) as excinfo:
        main(["ingest-etl4", "--config", str(config)])

    assert "does not exist" in str(excinfo.value)
    assert "etlcdb.db.aist.go.jp" in str(excinfo.value)


def test_train_vqvae_help_lists_its_flags(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["train-vqvae", "--help"])
    out = capsys.readouterr().out

    assert excinfo.value.code == 0
    for flag in ("--config", "--device", "--max-steps", "--output-root"):
        assert flag in out


def test_extract_codes_help_lists_its_flags(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["extract-codes", "--help"])
    out = capsys.readouterr().out

    assert excinfo.value.code == 0
    for flag in ("--config", "--checkpoint", "--split", "--out"):
        assert flag in out


def test_train_vqvae_runs_end_to_end(monkeypatch, tmp_path, tiny_config, tiny_data):
    monkeypatch.setattr("jvqvae.cli.load_run_config", lambda path: tiny_config)
    monkeypatch.setattr(
        "jvqvae.cli.load_splits", lambda config: {"train": tiny_data, "val": tiny_data}
    )

    code = main(
        [
            "train-vqvae",
            "--config",
            "unused.yaml",
            "--device",
            "cpu",
            "--max-steps",
            "2",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert code == 0
    assert (tmp_path / "checkpoints" / "tiny" / "vqvae_last.pt").exists()
    assert (tmp_path / "results" / "tiny" / "vqvae_run.json").exists()
    assert (tmp_path / "results" / "tiny" / "reconstructions.png").exists()


def test_extract_codes_writes_a_loadable_file(
    monkeypatch, tmp_path, tiny_config, tiny_data
):
    from jvqvae.codes import load_codes
    from jvqvae.models.vqvae import VQVAE, save_checkpoint

    checkpoint = tmp_path / "vqvae_best.pt"
    save_checkpoint(
        VQVAE(tiny_config.model),
        checkpoint,
        config=tiny_config.model,
        epoch=0,
        metrics={},
    )
    monkeypatch.setattr("jvqvae.cli.load_run_config", lambda path: tiny_config)
    monkeypatch.setattr("jvqvae.cli.load_splits", lambda config: {"train": tiny_data})

    out = tmp_path / "codes.npz"
    code = main(
        [
            "extract-codes",
            "--config",
            "unused.yaml",
            "--checkpoint",
            str(checkpoint),
            "--split",
            "train",
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
    )

    assert code == 0
    assert load_codes(out).codes.shape == (12, 4, 4)


def test_extract_codes_rejects_an_unknown_split(
    monkeypatch, tmp_path, tiny_config, tiny_data
):
    from jvqvae.models.vqvae import VQVAE, save_checkpoint

    checkpoint = tmp_path / "vqvae_best.pt"
    save_checkpoint(
        VQVAE(tiny_config.model),
        checkpoint,
        config=tiny_config.model,
        epoch=0,
        metrics={},
    )
    monkeypatch.setattr("jvqvae.cli.load_run_config", lambda path: tiny_config)
    monkeypatch.setattr("jvqvae.cli.load_splits", lambda config: {"train": tiny_data})

    with pytest.raises(SystemExit, match="train"):
        main(
            [
                "extract-codes",
                "--config",
                "unused.yaml",
                "--checkpoint",
                str(checkpoint),
                "--split",
                "nonsense",
                "--out",
                str(tmp_path / "codes.npz"),
            ]
        )


def test_train_prior_help_lists_its_flags(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["train-prior", "--help"])
    out = capsys.readouterr().out

    assert excinfo.value.code == 0
    for flag in ("--config", "--device", "--max-steps", "--output-root", "--codes"):
        assert flag in out


def test_sample_help_lists_its_flags(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["sample", "--help"])
    out = capsys.readouterr().out

    assert excinfo.value.code == 0
    for flag in (
        "--config",
        "--vqvae",
        "--prior",
        "--class",
        "--n",
        "--guidance",
        "--sweep-class",
    ):
        assert flag in out


def test_train_prior_runs_end_to_end(monkeypatch, tmp_path, tiny_config):
    import numpy as np

    from jvqvae.codes import CodeData, save_codes

    codes = CodeData(
        codes=np.zeros((8, 4, 4), dtype=np.int64),
        class_ids=np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64),
        classes=["あ", "い", "う"],
        num_codes=tiny_config.model.num_codes,
    )
    save_codes(codes, tmp_path / "results" / "tiny" / "codes_train.npz")
    monkeypatch.setattr("jvqvae.cli.load_run_config", lambda path: tiny_config)

    code = main(
        [
            "train-prior",
            "--config",
            "unused.yaml",
            "--device",
            "cpu",
            "--max-steps",
            "2",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert code == 0
    assert (tmp_path / "checkpoints" / "tiny" / "prior_last.pt").exists()
    assert (tmp_path / "results" / "tiny" / "prior_run.json").exists()


def test_sample_runs_end_to_end(monkeypatch, tmp_path, tiny_config):
    from jvqvae.models.prior import Prior, save_prior_checkpoint
    from jvqvae.models.vqvae import VQVAE, save_checkpoint

    vqvae_path = tmp_path / "vqvae.pt"
    prior_path = tmp_path / "prior.pt"
    save_checkpoint(
        VQVAE(tiny_config.model),
        vqvae_path,
        config=tiny_config.model,
        epoch=0,
        metrics={},
    )
    save_prior_checkpoint(
        Prior(
            tiny_config.prior,
            num_codes=tiny_config.model.num_codes,
            num_classes=3,
            latent_size=tiny_config.latent_size,
        ),
        prior_path,
        config=tiny_config.prior,
        num_codes=tiny_config.model.num_codes,
        num_classes=3,
        latent_size=tiny_config.latent_size,
        epoch=0,
        metrics={},
    )
    monkeypatch.setattr("jvqvae.cli.load_run_config", lambda path: tiny_config)

    code = main(
        [
            "sample",
            "--config",
            "unused.yaml",
            "--vqvae",
            str(vqvae_path),
            "--prior",
            str(prior_path),
            "--output-root",
            str(tmp_path),
            "--class",
            "0",
            "--n",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert code == 0
    assert (tmp_path / "results" / "tiny" / "samples.png").exists()


def test_train_prior_rejects_a_missing_codes_file(monkeypatch, tmp_path, tiny_config):
    monkeypatch.setattr("jvqvae.cli.load_run_config", lambda path: tiny_config)
    missing = tmp_path / "results" / "tiny" / "codes_train.npz"

    with pytest.raises(SystemExit, match="extract-codes") as excinfo:
        main(
            [
                "train-prior",
                "--config",
                "unused.yaml",
                "--output-root",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )

    assert str(missing) in str(excinfo.value)


def test_sample_rejects_a_missing_vqvae_checkpoint(monkeypatch, tmp_path, tiny_config):
    monkeypatch.setattr("jvqvae.cli.load_run_config", lambda path: tiny_config)
    missing = tmp_path / "checkpoints" / "tiny" / "vqvae_best.pt"

    with pytest.raises(SystemExit, match="train-vqvae") as excinfo:
        main(
            [
                "sample",
                "--config",
                "unused.yaml",
                "--output-root",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )

    assert str(missing) in str(excinfo.value)


def test_sample_rejects_a_missing_prior_checkpoint(monkeypatch, tmp_path, tiny_config):
    from jvqvae.models.vqvae import VQVAE, save_checkpoint

    vqvae_path = tmp_path / "checkpoints" / "tiny" / "vqvae_best.pt"
    save_checkpoint(
        VQVAE(tiny_config.model),
        vqvae_path,
        config=tiny_config.model,
        epoch=0,
        metrics={},
    )
    monkeypatch.setattr("jvqvae.cli.load_run_config", lambda path: tiny_config)
    missing = tmp_path / "checkpoints" / "tiny" / "prior_best.pt"

    with pytest.raises(SystemExit, match="train-prior") as excinfo:
        main(
            [
                "sample",
                "--config",
                "unused.yaml",
                "--output-root",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )

    assert str(missing) in str(excinfo.value)
