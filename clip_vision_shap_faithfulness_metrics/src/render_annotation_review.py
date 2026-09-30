"""Render the pointing-game boxes on the original images for human review.

The review images deliberately show no SHAP information, so the boxes are checked
against the image content only. The dashed frame marks the centre crop CLIP sees;
only the part of a box inside that frame can ever be pointed at.

Edit annotations/pointing_game_boxes.csv (coordinates in original image pixels,
x0 y0 = top-left, x1 y1 = bottom-right) and re-run this script to update the review.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "annotations" / "pointing_game_boxes.csv"
REVIEW_DIR = ROOT / "annotations" / "review"
IMAGES_DIR = ROOT.parent / "clip_vision_images" / "images"
SUMMARY = ROOT.parent / "clip_vision_shap" / "outputs" / "values" / "clip_vision_i1_i10" / "image_summary.csv"
BOX = "#ffd400"
CROP = "#00e5ff"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def draw(axis: plt.Axes, image_id: str, boxes: list[dict[str, str]], crop: list[float], title_size: int) -> None:
    image = Image.open(IMAGES_DIR / f"{image_id}.jpg").convert("RGB")
    axis.imshow(image)
    left, top, right, bottom = crop
    axis.add_patch(patches.Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor=CROP, linewidth=1.6, linestyle=(0, (5, 3))))
    for box in boxes:
        x0, y0, x1, y1 = (float(box[key]) for key in ("x0", "y0", "x1", "y1"))
        for width, colour in ((4.0, "black"), (2.2, BOX)):
            axis.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colour, linewidth=width))
        axis.text(x0 + 3, y0 + 3, f"{box['box_id']}: {box['target_object']}", fontsize=title_size - 3, color="black", va="top",
                  bbox={"facecolor": BOX, "edgecolor": "none", "pad": 1.5})
    status = sorted({box["status"] for box in boxes})
    axis.set_title(f"{image_id} - target: {boxes[0]['target_object']}  [{', '.join(status)}]", fontsize=title_size, loc="left")
    axis.set_axis_off()


def main() -> None:
    boxes = read_csv(ANNOTATIONS)
    crops = {row["image_id"]: [float(v) for v in row["crop_box_original"].split(";")] for row in read_csv(SUMMARY)}
    by_image: dict[str, list[dict[str, str]]] = {}
    for box in boxes:
        by_image.setdefault(box["image_id"], []).append(box)
    order = [f"I{index}" for index in range(1, 11)]
    missing = [image_id for image_id in order if image_id not in by_image]
    if missing:
        raise ValueError(f"No box for {missing}")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for image_id in order:
        figure, axis = plt.subplots(figsize=(8, 6))
        draw(axis, image_id, by_image[image_id], crops[image_id], 13)
        notes = "\n".join(f"box {box['box_id']}: {box['note']}  (x0={box['x0']}, y0={box['y0']}, x1={box['x1']}, y1={box['y1']})" for box in by_image[image_id])
        figure.text(0.01, 0.01, notes + "\nyellow = draft box (original-image pixels); dashed cyan = centre crop CLIP sees",
                    fontsize=8.5, va="bottom", color="#333333")
        figure.savefig(REVIEW_DIR / f"{image_id}_review.png", dpi=130, bbox_inches="tight")
        plt.close(figure)
    figure, axes = plt.subplots(2, 5, figsize=(22, 8.5))
    for axis, image_id in zip(axes.reshape(-1), order):
        draw(axis, image_id, by_image[image_id], crops[image_id], 11)
    figure.suptitle("Pointing game - boxes for review (yellow = box, dashed cyan = CLIP centre crop)", fontsize=14, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(REVIEW_DIR / "all_images_review.png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {len(order) + 1} review images to {REVIEW_DIR}")


if __name__ == "__main__":
    main()
