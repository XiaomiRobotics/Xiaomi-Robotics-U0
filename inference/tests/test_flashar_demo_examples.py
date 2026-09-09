from __future__ import annotations

import importlib.util
import unittest

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

if HAS_TRANSFORMERS:
    from xr_u0_flashar.demo_examples import ROOT, demo_example_for, example_choices, flashar_demo_examples
    from xr_u0_flashar.demo_runtime import SUPPORTED_TASKS, TASK_SCENE, TASK_T2I, TASK_TRANSFER, TASK_X2I


@unittest.skipUnless(HAS_TRANSFORMERS, "FlashAR demo imports require transformers")
class FlashARDemoExamplesTest(unittest.TestCase):
    def test_each_supported_task_has_examples(self):
        examples = flashar_demo_examples()
        for task in SUPPORTED_TASKS:
            with self.subTest(task=task):
                self.assertIn(task, examples)
                self.assertGreater(len(examples[task]), 0)
                self.assertEqual(example_choices(task), [item.id for item in examples[task]])

    def test_reference_examples_point_to_repo_assets(self):
        examples = flashar_demo_examples()
        for task in (TASK_X2I, TASK_TRANSFER):
            for example in examples[task]:
                with self.subTest(task=task, example=example.id):
                    self.assertIsNotNone(example.reference_image)
                    self.assertTrue(example.reference_image.startswith("examples/assets/"))
                    self.assertTrue((ROOT / example.reference_image).is_file())
                    self.assertTrue(example.reference_path)

        for example in examples[TASK_T2I]:
            self.assertIsNone(example.reference_image)
            self.assertIsNone(example.reference_path)

    def test_scene_and_transfer_prompts_are_ui_text_only(self):
        examples = flashar_demo_examples()
        for task in (TASK_SCENE, TASK_TRANSFER):
            for example in examples[task]:
                with self.subTest(task=task, example=example.id):
                    self.assertTrue(example.prompt.startswith("Workspace:"))
                    self.assertNotIn("You are an advanced", example.prompt)
                    self.assertNotIn("Scene Description:", example.prompt)
                    self.assertNotIn("<|", example.prompt)

    def test_scene_metadata_is_extracted(self):
        example = demo_example_for(TASK_SCENE, "scene_printing_press")
        self.assertEqual(example.robot_arm_type, "ARX Robot")
        self.assertEqual(example.image_style, "Real")
        self.assertEqual(example.cfg_scale, 2.0)
        self.assertEqual((example.height, example.width), (32, 128))

    def test_transfer_examples_default_to_depth_input(self):
        example = demo_example_for(TASK_TRANSFER, "transfer_sort_fruit")
        self.assertEqual(example.input_image_type, "depth")
        self.assertEqual(example.image_style, "Simulator")
        self.assertEqual((example.height, example.width), (0, 0))


if __name__ == "__main__":
    unittest.main()
