# Example Assets

This directory contains small public reference assets used by runnable examples.

- `x2i/`: Single-image references used by the X2I task examples in
  `configs/tasks/x2i.py`. `edit_reference.png` is kept as a simple generic
  image-editing reference.
- `video-gen/`: Initial observations used by Video Gen examples in
  `configs/tasks/video_gen.py`.
- `transfer/`: Transfer input examples. Depth files are horizontal triptychs
  with three `1024x1024` views in `head | hand_left | hand_right` order, for a
  total size of `3072x1024`.
- `transfer/*_rgb_triptych.png`: RGB reference triptychs for visually checking
  the matching Transfer examples or testing RGB-to-depth preprocessing.
- `transfer/legacy_depth_triptych.png`: Depth input for the restored
  `transfer_lemon_depth` example, kept under the legacy filename to preserve
  compatibility with earlier local commands.
- `transfer/legacy_rgb_triptych_gt.png`: Matching legacy RGB target/reference
  asset for visual checks and RGB-to-depth preprocessing tests.
