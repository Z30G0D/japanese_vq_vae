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
    assert "fetch-k49" in capsys.readouterr().out


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
