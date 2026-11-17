"""Reject substituted archives before invoking SB3's deserializer."""

import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "trace_training", ROOT / "trace_training.py"
)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def test_rejects_substitution_before_deserialization(tmp_path, monkeypatch):
    checkpoint = tmp_path / "runs/seed-7/final.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"untrusted archive")
    loader = Mock()
    monkeypatch.setattr(trace, "ROOT", tmp_path)
    monkeypatch.setattr(trace.PPO, "load", loader)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        trace.load_bundled_checkpoint(None)
    loader.assert_not_called()


def test_ignores_cwd_checkpoint_and_loads_verified_bytes(tmp_path, monkeypatch):
    impostor = tmp_path / "runs/seed-7/final.zip"
    impostor.parent.mkdir(parents=True)
    impostor.write_bytes(b"untrusted cwd archive")
    monkeypatch.chdir(tmp_path)
    expected = (ROOT / "runs/seed-7/final.zip").read_bytes()
    loader = Mock(return_value="model")
    monkeypatch.setattr(trace.PPO, "load", loader)
    model, digest = trace.load_bundled_checkpoint(None)
    assert model == "model"
    assert digest == trace.CHECKPOINT_SHA256
    stream = loader.call_args.args[0]
    assert stream.read() == expected
    assert hashlib.sha256(expected).hexdigest() == digest
