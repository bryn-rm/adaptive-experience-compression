import logging

import pytest
from pydantic import ValidationError

from aec.cli import main
from aec.config import Config, load_config
from aec.logging import configure_logging


def test_default_config():
    assert load_config("configs/default.yaml") == Config()


@pytest.mark.parametrize("content", ["", "- INFO", "log_level: NOPE", "unknown: 1", "log_level: [INFO]"])
def test_invalid_config_rejected(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    with pytest.raises((ValueError, ValidationError)):
        load_config(path)


def test_missing_config_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_cli(caplog):
    with caplog.at_level(logging.INFO):
        assert main(["--config", "configs/default.yaml"]) == 0
    assert "Configuration validated" in caplog.text


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_logging_configuration(monkeypatch):
    calls = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: calls.append(kwargs))
    configure_logging("DEBUG")
    assert calls[0]["level"] == "DEBUG"
