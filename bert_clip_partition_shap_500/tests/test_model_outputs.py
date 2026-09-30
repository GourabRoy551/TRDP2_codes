"""Output contract [NEG, POS, POS-NEG], label mapping, parity and the no-training guarantee."""

from __future__ import annotations

import re

import numpy as np
import pytest

from io_utils import CHECKPOINT_DIR, ROOT, load_json
from model_outputs import OUTPUT_NAMES, assemble_outputs, predict_label


def test_output_order_is_neg_pos_margin():
    assert OUTPUT_NAMES == ("NEG", "POS", "POS-NEG")
    outputs = assemble_outputs(np.asarray([[1.5, -0.5], [0.25, 0.75]], dtype=np.float32))
    assert outputs.shape == (2, 3)
    np.testing.assert_array_equal(outputs[:, 0], [1.5, 0.25])
    np.testing.assert_array_equal(outputs[:, 1], [-0.5, 0.75])
    np.testing.assert_array_equal(outputs[:, 2], outputs[:, 1] - outputs[:, 0])


def test_prediction_rule_margin_strictly_positive_is_pos():
    assert predict_label(0.0) == "NEG"
    assert predict_label(-1e-9) == "NEG"
    assert predict_label(1e-9) == "POS"


@pytest.mark.parametrize("model", ["bert", "clip"])
def test_saved_margin_equals_pos_minus_neg(model, model_scores, sentence_results):
    for table, pos_key, neg_key in ((model_scores, "pos_score", "neg_score"), (sentence_results, "pos_score", "neg_score")):
        selected = [row for row in table if row["model"] == model]
        assert len(selected) == 500
        for row in selected:
            assert float(row["margin"]) == float(row[pos_key]) - float(row[neg_key])
            assert row["prediction"] == ("POS" if float(row["margin"]) > 0 else "NEG")


def test_model_checks_gate(config):
    checks = load_json(CHECKPOINT_DIR / "model_checks.json")
    assert checks["gate_passed"] is True
    bert = checks["models"]["bert"]
    assert bert["label_mapping"]["NEG"] == 0 and bert["label_mapping"]["POS"] == 1
    assert bert["label_mapping_verification"]["correct_under_mapping"] == 10
    assert bert["label_mapping_verification"]["correct_under_swapped_mapping"] == 0
    assert checks["models"]["clip"]["prompt_source_verification"]["passed"] is True
    for model in ("bert", "clip"):
        parity = checks["models"][model]["parity"]
        assert parity["sentences_checked"] >= 10
        assert parity["output_shape"] == [parity["sentences_checked"], 3]
        assert parity["passed"] is True
        assert checks["models"][model]["frozen"]["trainable_parameter_tensors"] == 0


@pytest.fixture(scope="module")
def live_device():
    torch = pytest.importorskip("torch")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_live_bert_columns_are_raw_neg_pos_logits(config, rows, live_device):
    import torch

    from bert_backend import BertThreeOutputScorer, load_bert, resolve_label_mapping

    model, tokenizer = load_bert(config, live_device)
    mapping = resolve_label_mapping(model, config)
    scorer = BertThreeOutputScorer(model, tokenizer, live_device, config["models"]["bert"]["max_length"], mapping)
    texts = [row["text"] for row in rows[:12]]
    outputs = scorer(texts)
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(live_device)
    with torch.inference_mode():
        logits = model(**encoded).logits.float().cpu().numpy().astype(np.float64)
    np.testing.assert_array_equal(outputs[:, 0], logits[:, 0])
    np.testing.assert_array_equal(outputs[:, 1], logits[:, 1])
    np.testing.assert_array_equal(outputs[:, 2], outputs[:, 1] - outputs[:, 0])
    assert not any(parameter.requires_grad for parameter in model.parameters()) and not model.training


def test_live_clip_columns_are_prototype_similarities(config, rows, live_device):
    import torch

    from clip_backend import create_clip_scorer, encode_texts

    scorer, info = create_clip_scorer(config, live_device)
    texts = [row["text"] for row in rows[:12]]
    outputs = scorer(texts)
    embeddings = encode_texts(scorer.model, scorer.tokenizer, texts, live_device, 77)
    expected = (embeddings @ scorer.prototypes.T).float().cpu().numpy().astype(np.float64)
    np.testing.assert_array_equal(outputs[:, :2], expected)
    np.testing.assert_array_equal(outputs[:, 2], outputs[:, 1] - outputs[:, 0])
    np.testing.assert_allclose(info["prototype_norms"], 1.0, atol=1e-6)
    assert not any(parameter.requires_grad for parameter in scorer.model.parameters())
    assert torch.is_tensor(scorer.prototypes)


FORBIDDEN = [
    r"\.backward\(",
    r"torch\.optim",
    r"optimizer",
    r"zero_grad",
    r"lr_scheduler",
    r"\bTrainer\b",
    r"TrainingArguments",
    r"\.train\(\s*\)",
    r"requires_grad_\(\s*True",
    r"requires_grad\s*=\s*True",
    r"\.fit\(",
    r"save_pretrained",
    r"LogisticRegression",
]


def test_experiment_contains_no_training_or_fine_tuning_code():
    offenders = []
    for path in sorted((ROOT / "src").glob("*.py")) + [ROOT / "run_experiment.bat"]:
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN:
            if re.search(pattern, text):
                offenders.append(f"{path.name}: {pattern}")
    assert offenders == []
