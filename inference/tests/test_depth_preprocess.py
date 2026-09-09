from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from xr_u0_ar.depth_preprocess import (
    DA3_DEPTH_ENCODING,
    DA3_TRANSFER_VIEW_COUNT,
    DEFAULT_DA3_MODEL_PATH,
    DepthPreprocessOptions,
    depth_options_from_config,
    normalize_inverse_depth_dynamic,
    normalize_inverse_depth,
    rgb_to_inverse_depth_image,
    validate_depth_preprocess_config,
)


class FakePrediction:
    def __init__(self, depth):
        self.depth = depth


class FakeDA3:
    def __init__(self, depths=None):
        self.depths = depths
        self.calls = []

    def inference(self, *, image, process_res):
        self.calls.append(
            {
                "process_res": process_res,
                "sizes": [item.size for item in image],
            }
        )
        if self.depths is not None:
            return FakePrediction(self.depths)
        depths = []
        for view in image:
            width, height = view.size
            depths.append(np.linspace(0.7, 2.0, width * height, dtype=np.float32).reshape(height, width))
        return FakePrediction(depths)


class DepthPreprocessTest(unittest.TestCase):
    def test_inverse_depth_is_bright_near(self):
        depth = np.array([[0.7, 2.0]], dtype=np.float32)
        norm = normalize_inverse_depth(depth, min_depth=0.7, max_depth=2.0)
        self.assertGreater(norm[0, 0], norm[0, 1])
        self.assertEqual(float(norm[0, 0]), 1.0)
        self.assertEqual(float(norm[0, 1]), 0.0)

    def test_dynamic_inverse_depth_is_per_image_bright_near(self):
        depth = np.array([[10.0, 40.0]], dtype=np.float32)
        gray, stats = normalize_inverse_depth_dynamic(depth)
        self.assertGreater(gray[0, 0], gray[0, 1])
        self.assertEqual(int(gray[0, 0]), 255)
        self.assertEqual(int(gray[0, 1]), 0)
        self.assertEqual(stats["depth_min"], 10.0)
        self.assertEqual(stats["depth_max"], 40.0)

    def test_rgb_to_inverse_depth_image_splits_triptych_and_outputs_rgb_grayscale(self):
        image = Image.new("RGB", (7, 2), color=(10, 20, 30))
        model = FakeDA3()
        out = rgb_to_inverse_depth_image(
            image,
            model=model,
            min_depth=0.7,
            max_depth=2.0,
            process_res=504,
            device="cpu",
        )
        arr = np.asarray(out)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(arr.shape, (2, 7, 3))
        self.assertTrue(np.array_equal(arr[..., 0], arr[..., 1]))
        self.assertTrue(np.array_equal(arr[..., 1], arr[..., 2]))
        self.assertEqual(model.calls, [{"process_res": 504, "sizes": [(2, 2), (2, 2), (3, 2)]}])
        self.assertGreater(arr[0, 0, 0], arr[0, 1, 0])
        self.assertGreater(arr[0, 2, 0], arr[0, 3, 0])
        self.assertGreater(arr[0, 4, 0], arr[0, 6, 0])

    def test_rgb_to_inverse_depth_image_uses_per_view_dynamic_minmax(self):
        image = Image.new("RGB", (6, 1), color=(10, 20, 30))
        model = FakeDA3(
            depths=[
                np.array([[1.0, 4.0]], dtype=np.float32),
                np.array([[10.0, 40.0]], dtype=np.float32),
                np.array([[100.0, 400.0]], dtype=np.float32),
            ]
        )
        out = rgb_to_inverse_depth_image(
            image,
            model=model,
            min_depth=0.7,
            max_depth=2.0,
            process_res=504,
            device="cpu",
        )
        arr = np.asarray(out)[0, :, 0]
        self.assertEqual(arr.tolist(), [255, 0, 255, 0, 255, 0])

    def test_rgb_to_inverse_depth_image_writes_artifacts(self):
        image = Image.new("RGB", (6, 2), color=(10, 20, 30))
        options = DepthPreprocessOptions(
            input_image_type="rgb",
            da3_model_path="local/DA3-LARGE-1.1",
            da3_device="cpu",
            da3_process_res=504,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = Path(tmpdir)
            out = rgb_to_inverse_depth_image(
                image,
                model=FakeDA3(),
                min_depth=0.7,
                max_depth=2.0,
                process_res=504,
                device="cpu",
                artifact_dir=tmpdir,
                source_path="input.png",
                options=options,
            )
            metadata = json.loads((artifact_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_path"], "input.png")
            self.assertEqual(metadata["da3_model_path"], "local/DA3-LARGE-1.1")
            self.assertEqual(metadata["da3_depth_encoding"], DA3_DEPTH_ENCODING)
            self.assertEqual(metadata["da3_view_count"], DA3_TRANSFER_VIEW_COUNT)
            self.assertEqual(metadata["output_size"], [out.width, out.height])
            self.assertEqual(len(metadata["views"]), 3)
            for name in (
                "reference_rgb.png",
                "depth_triptych.png",
                "view_0_rgb.png",
                "view_0_depth.png",
                "view_1_rgb.png",
                "view_1_depth.png",
                "view_2_rgb.png",
                "view_2_depth.png",
                "metadata.json",
            ):
                with self.subTest(name=name):
                    self.assertTrue((artifact_root / name).exists())

    def test_rgb_transfer_defaults_to_large_da3_model_path(self):
        cfg = SimpleNamespace(
            task_type="Transfer",
            input_image_type="rgb",
            da3_model_path=None,
            prompts={"sample": {"task_type": "Transfer"}},
        )
        validate_depth_preprocess_config(cfg)
        options = depth_options_from_config(cfg)
        self.assertEqual(options.da3_model_path, DEFAULT_DA3_MODEL_PATH)

    def test_depth_transfer_does_not_require_da3_model_path(self):
        cfg = SimpleNamespace(
            task_type="Transfer",
            input_image_type="depth",
            da3_model_path=None,
            prompts={"sample": {"task_type": "Transfer"}},
        )
        validate_depth_preprocess_config(cfg)

    def test_options_accept_depth_default(self):
        opts = DepthPreprocessOptions()
        self.assertEqual(opts.input_image_type, "depth")


if __name__ == "__main__":
    unittest.main()
