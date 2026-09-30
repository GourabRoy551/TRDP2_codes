# Pointing-game boxes for I1–I10 (approved)

The images have no dataset annotations, so the boxes here were drafted for the pointing game
and then checked by the author. The author approved them on 2026-09-30, unchanged from the
draft, and no image was excluded. The report describes them as author-annotated, not as
benchmark ground truth.

- `pointing_game_boxes.csv`: one row per box, in original-image pixels (`x0 y0` = top-left,
  `x1 y1` = bottom-right). An image can have several boxes of the same target, such as the
  two horses in I1.
- `review/I<n>_review.png`: each image with its box(es) and coordinates.
- `review/all_images_review.png`: all ten on one sheet.
- The dashed cyan frame is the 224×224 centre crop that CLIP sees. Only the part of a box
  inside it can be hit.

The drafts were made while looking at the images only. The SHAP maps were not shown, so the
boxes are independent of the explanation.

## To approve or correct

1. **Approve:** open each review image. If a box is right, change its `status` from `draft`
   to `approved`.
2. **Correct:** edit `target_object` and/or `x0 y0 x1 y1`, or add or delete rows. Then run
   `%USERPROFILE%\miniconda3\envs\rfem\python.exe src\render_annotation_review.py` to redraw.
3. **Exclude:** to leave an image out (for example a pure landscape), set its `status` to `excluded`.

Only rows with `status = approved` will be used. The metric will not run while any row is still `draft`.
