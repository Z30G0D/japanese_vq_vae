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

    assert captured["cache"] == f"{tmp_path}/etl4_64.npz"
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
