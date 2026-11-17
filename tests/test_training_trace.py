"""Check exported targets against experience and loss against frozen targets."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_trace_provenance_and_targets():
    trace = json.loads((ROOT / "artifacts/training-trace.json").read_text())
    assert (
        trace["checkpoint_sha256"]
        == hashlib.sha256((ROOT / "runs/seed-7/final.zip").read_bytes()).hexdigest()
    )
    b = {k: np.array(v) for k, v in trace["buffer"].items()}
    assert b["observations"].shape == (512, 4, 9)
    np.testing.assert_allclose(b["returns"], b["values"] + b["advantages"], atol=1e-6)
    # All internal transitions, including boundaries. Stored timeout rewards
    # already contain terminal-value bootstrapping; never cross into reset state.
    mask = 1 - b["episode_starts"][1:]
    delta = b["rewards"][:-1] + 0.99 * b["values"][1:] * mask - b["values"][:-1]
    expected = delta + 0.99 * 0.95 * mask * b["advantages"][1:]
    np.testing.assert_allclose(b["advantages"][:-1], expected, atol=3e-6)
    last = (
        b["rewards"][-1]
        + 0.99
        * np.array(trace["bootstrap_values"])
        * (1 - np.array(trace["last_dones"], dtype=int))
        - b["values"][-1]
    )
    np.testing.assert_allclose(b["advantages"][-1], last, atol=1e-6)
    assert any(any(t["timeouts"]) for t in trace["transitions"])


def test_losses_and_selected_row():
    trace = json.loads((ROOT / "artifacts/training-trace.json").read_text())
    summary = json.loads((ROOT / "artifacts/training-example.json").read_text())
    batch = trace["minibatch"]
    a = np.array(batch["advantages"])
    a = (a - a.mean()) / (a.std(ddof=1) + 1e-8)
    np.testing.assert_allclose(a, batch["normalized_advantages"], atol=1e-6)
    for phase in ("before", "after"):
        out = {k: np.array(v) for k, v in trace[phase].items()}
        ratio = np.exp(out["log_prob"] - batch["old_log_prob"])
        policy = -np.minimum(ratio * a, np.clip(ratio, 0.8, 1.2) * a)
        value = (out["value"] - batch["returns"]) ** 2
        np.testing.assert_allclose(out["policy_loss"], policy, atol=1e-6)
        np.testing.assert_allclose(out["value_loss"], value, atol=3e-6)
        np.testing.assert_allclose(
            out["total_loss"],
            policy.mean() + 0.5 * value.mean() - 0.005 * out["entropy"].mean(),
            atol=1e-6,
        )
        assert summary[phase]["value"] == trace[phase]["value"][0]
    r = summary["row"]
    assert (
        r["observations"]
        == trace["buffer"]["observations"][r["step"]][r["environment"]]
    )
    assert r["observations"] == batch["observations"][0]
    assert summary["before"]["value"] != summary["after"]["value"]
