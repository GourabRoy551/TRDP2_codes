"""Copy the report materials into this folder and write MANIFEST.csv.

Sources are only read. Every copy is verified by SHA-256 against its source, and the
run stops if an expected file is missing. Re-running replaces the copies with the
current source files.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRDP2 = HERE.parent

METRICS = TRDP2 / "bert_clip_shap_faithfulness_metrics"
VISION_METRICS = TRDP2 / "clip_vision_shap_faithfulness_metrics"
DUAL_BERT_PLOTS = TRDP2 / "bert_shap" / "dual_class_bert" / "outputs" / "plots" / "bert_dual_20_curated"
CLIP_TEXT_PLOTS = TRDP2 / "clip_text_shap" / "outputs" / "plots" / "clip_text_dual_20_curated"
CLIP_VISION_PLOTS = TRDP2 / "clip_vision_shap" / "outputs" / "plots" / "clip_vision_i1_i10"

SENTENCES = [f"S{index}" for index in range(1, 11)]
IMAGES = [f"I{index}" for index in range(1, 11)]
SENTENCE_PLOTS = ["dual_class_word_bar", "presentation_overview", "summary_matrix", "token_class_matrix", "word_class_matrix"]
METRIC_FILES = [
    "comparison_table.md",
    "comparison_table.csv",
    "comparison_table.tex",
    "per_sentence_metrics.csv",
    "aggregate_by_fraction.csv",
    "perturbation_audit.csv",
    "rationale_reference_words.csv",
    "checks.json",
]
VISION_METRIC_FILES = [
    "summary_table.md",
    "summary_table.csv",
    "summary_table.tex",
    "per_image_metrics.csv",
    "curve_points.csv",
    "checks.json",
    "plots/deletion_insertion_curves_mean.png",
    "plots/deletion_insertion_curves_mean.pdf",
    "plots/deletion_insertion_curves_per_image.png",
    "plots/deletion_insertion_curves_per_image.pdf",
    "plots/pointing_game.png",
    "plots/pointing_game.pdf",
]
VISION_ANNOTATION_FILES = {  # source (under annotations/) -> name in the report folder
    "pointing_game_boxes.csv": "pointing_game_boxes.csv",
    "review/all_images_review.png": "pointing_game_boxes_review.png",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planned_copies() -> list[tuple[str, Path, Path]]:
    items: list[tuple[str, Path, Path]] = []
    target = HERE / "01_bert_clip_faithfulness_metrics"
    for name in METRIC_FILES:
        items.append(("bert_clip_faithfulness_metrics", METRICS / "outputs" / name, target / name))
    items.append(("bert_clip_faithfulness_metrics", METRICS / "README.md", target / "EXPERIMENT_README.md"))
    for group, source_root, folder in (
        ("dual_class_bert", DUAL_BERT_PLOTS, "02_dual_class_bert_S1_S10"),
        ("clip_text_shap", CLIP_TEXT_PLOTS, "03_clip_text_shap_S1_S10"),
    ):
        for sentence in SENTENCES:
            for plot in SENTENCE_PLOTS:
                for extension in ("png", "pdf"):
                    name = f"{sentence}_{plot}.{extension}"
                    items.append((group, source_root / sentence / name, HERE / folder / sentence / name))
    for image in IMAGES:
        for extension in ("png", "pdf"):
            name = f"{image}_SHAP_patch_overlay_values.{extension}"
            items.append(("clip_vision_shap", CLIP_VISION_PLOTS / image / name, HERE / "04_clip_vision_shap_I1_I10_patch_overlay_values" / name))
    target = HERE / "05_clip_vision_faithfulness_metrics"
    for name in VISION_METRIC_FILES:
        items.append(("clip_vision_faithfulness_metrics", VISION_METRICS / "outputs" / name, target / Path(name).name))
    for source, name in VISION_ANNOTATION_FILES.items():
        items.append(("clip_vision_faithfulness_metrics", VISION_METRICS / "annotations" / source, target / name))
    items.append(("clip_vision_faithfulness_metrics", VISION_METRICS / "README.md", target / "EXPERIMENT_README.md"))
    return items


def main() -> None:
    items = planned_copies()
    missing = [str(source) for _, source, _ in items if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} expected source file(s) missing, e.g. {missing[:3]}")
    for sentence_root in (DUAL_BERT_PLOTS, CLIP_TEXT_PLOTS):
        for sentence in SENTENCES:
            extra = {path.name for path in (sentence_root / sentence).iterdir()} - {
                f"{sentence}_{plot}.{extension}" for plot in SENTENCE_PLOTS for extension in ("png", "pdf")
            }
            if extra:
                raise RuntimeError(f"Unexpected plot files in {sentence_root / sentence}: {sorted(extra)}")
    rows = []
    for group, source, destination in items:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256(source)
        if sha256(destination) != digest:
            raise RuntimeError(f"Copy verification failed for {destination}")
        rows.append(
            {
                "group": group,
                "report_file": destination.relative_to(HERE).as_posix(),
                "source_file": source.relative_to(TRDP2).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": digest,
            }
        )
    with (HERE / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    print(f"copied and verified {len(rows)} files: {counts}")


if __name__ == "__main__":
    main()
