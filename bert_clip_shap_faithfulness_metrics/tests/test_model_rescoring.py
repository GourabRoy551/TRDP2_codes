"""Stored sufficiency scores are reproducible under a different batch composition."""

from __future__ import annotations

import numpy as np
import pytest

from model_outputs import MARGIN


@pytest.mark.parametrize("model_key", ["bert", "clip"])
def test_sufficiency_margins_rescore_individually(model_key, audit, source_config):
    from run_experiment import _release_memory, build_scorer, setup

    rows = [row for row in audit if row["model"] == model_key][:40]
    device = setup(source_config)
    try:
        scorer, _ = build_scorer(model_key, source_config, device)
    except OSError as error:  # pragma: no cover - only when the local model cache is missing
        pytest.skip(f"local model unavailable: {error}")
    single = np.asarray([scorer([row["sufficiency_text"]])[0, MARGIN] for row in rows])
    del scorer
    _release_memory()
    stored = np.asarray([float(row["sufficiency_margin"]) for row in rows])
    assert np.max(np.abs(single - stored)) <= float(source_config["tolerances"]["batch_parity"][model_key])
