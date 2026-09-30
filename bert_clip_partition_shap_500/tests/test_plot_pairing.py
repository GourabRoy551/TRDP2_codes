"""Exactly five aggregate and five case-study figures, each as a PNG + PDF pair."""

from __future__ import annotations

from io_utils import CASE_STUDY_DIR, PLOTS_DIR, VALUES_DIR, read_csv


def stems(directory, suffix):
    return {path.stem for path in directory.glob(f"*{suffix}")}


def test_every_png_has_matching_pdf():
    for directory in (PLOTS_DIR, CASE_STUDY_DIR):
        assert stems(directory, ".png") == stems(directory, ".pdf")


def test_exactly_five_aggregate_plot_pairs():
    assert len(stems(PLOTS_DIR, ".png")) == 5
    assert sorted(stems(PLOTS_DIR, ".png")) == [
        "fig1_model_performance", "fig2_additivity", "fig3_faithfulness", "fig4_bert_clip_agreement", "fig5_efficiency"]


def test_exactly_five_case_study_pairs_for_the_selected_sentences():
    names = stems(CASE_STUDY_DIR, ".png")
    assert len(names) == 5
    selected = [row["sentence_id"] for row in read_csv(VALUES_DIR / "representative_selection.csv")]
    assert all(any(sentence_id in name for name in names) for sentence_id in selected)


def test_word_bar_figures_are_paired_and_additive(config):
    root = PLOTS_DIR / "word_bars"
    ours = root / "bert_clip_partition_shap_500"
    assert len(stems(ours, ".png")) == 8
    for directory in [path for path in root.iterdir() if path.is_dir()]:
        assert stems(directory, ".png") == stems(directory, ".pdf")
    manifest = read_csv(VALUES_DIR / "word_bar_plot_manifest.csv")
    assert len(manifest) == sum(len(stems(d, ".png")) for d in root.iterdir() if d.is_dir())
    for row in manifest:
        if row["experiment"] == "bert_clip_partition_shap_500":
            tolerance = float(config["tolerances"]["additivity"][row["model_or_condition"]])
            assert max(float(row[f"{k}_word_level_residual"]) for k in ("neg", "pos", "margin")) <= tolerance


def test_plot_files_are_non_empty():
    for directory in (PLOTS_DIR, CASE_STUDY_DIR):
        for path in list(directory.glob("*.png")) + list(directory.glob("*.pdf")):
            assert path.stat().st_size > 10_000, path.name
