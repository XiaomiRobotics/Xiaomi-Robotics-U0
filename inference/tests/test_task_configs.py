from __future__ import annotations

import unittest

from configs.compose import compose_dict


class TaskConfigTest(unittest.TestCase):
    def test_ar_defaults_to_34b_checkpoint(self):
        cfg = compose_dict(engine="ar", backend="eager", task="t2i", num_samples=1)
        self.assertEqual(cfg["model_size"], "34b")
        self.assertEqual(cfg["model_path"], "checkpoints/Xiaomi-Robotics-U0")

    def test_4b_checkpoint_is_ar_only(self):
        cfg = compose_dict(engine="ar", backend="eager", task="x2i", model_size="4b", num_samples=1)
        self.assertEqual(cfg["model_size"], "4b")
        self.assertEqual(cfg["model_path"], "../ckpt/Xiaomi-Robotics-U0-4B")
        with self.assertRaisesRegex(ValueError, "AR-only"):
            compose_dict(engine="flashar", backend="vllm", task="t2i", model_size="4b", num_samples=1)

    def test_old_sequence_names_are_removed(self):
        for task in ("video-gen", "video_gen", "interleave"):
            with self.assertRaisesRegex(ValueError, "unsupported task"):
                compose_dict(engine="ar", backend="eager", task=task, model_size="4b")

    def test_default_uses_active_examples(self):
        cfg = compose_dict(engine="ar", backend="eager", task="t2i")
        self.assertEqual(
            list(cfg["prompts"]),
            [
                "t2i_glass_teapot",
                "t2i_workshop_camera",
                "t2i_robotic_gripper",
                "t2i_ceramic_vase",
                "t2i_lab_sample",
            ],
        )

    def test_prompt_override_uses_num_samples(self):
        cfg = compose_dict(
            engine="ar",
            backend="eager",
            task="t2i",
            prompt="A red cube on a table.",
            num_samples=3,
        )
        self.assertEqual(list(cfg["prompts"]), ["t2i_000", "t2i_001", "t2i_002"])
        self.assertEqual(cfg["prompts"]["t2i_001"]["prompt"], "A red cube on a table.")
        self.assertEqual(cfg["prompts"]["t2i_001"]["id"], "t2i_001")

    def test_default_examples_use_num_samples_as_limit(self):
        cfg = compose_dict(engine="flashar", backend="vllm", task="t2i", num_samples=1)
        self.assertEqual(list(cfg["prompts"]), ["t2i_glass_teapot"])

    def test_ar_eager_defaults_to_dependency_free_attention(self):
        cfg = compose_dict(engine="ar", backend="eager", task="t2i", num_samples=1)
        self.assertEqual(cfg["attn_implementation"], "eager")

    def test_default_reference_examples_are_repo_relative(self):
        cfg = compose_dict(engine="flashar", backend="eager", task="x2i")
        self.assertEqual(len(cfg["prompts"]), 5)
        for case in cfg["prompts"].values():
            self.assertTrue(case["reference_image"].startswith("examples/assets/x2i/"))

    def test_scene_gen_includes_printing_press_case(self):
        cfg = compose_dict(engine="flashar", backend="vllm", task="scene-gen")
        self.assertEqual(
            list(cfg["prompts"]),
            ["scene_floral_foam", "scene_printing_press", "scene_soviet_spacecraft"],
        )
        case = cfg["prompts"]["scene_printing_press"]
        self.assertIn("Robot Arm Type: ARX Robot.", case["text_prompt"])
        self.assertIn("wooden printing press bed", case["text_prompt"])

    def test_transfer_includes_all_depth_examples(self):
        cfg = compose_dict(engine="flashar", backend="vllm", task="transfer")
        self.assertEqual(
            list(cfg["prompts"]),
            ["transfer_towel", "transfer_sort_fruit", "transfer_lemon_depth"],
        )
        self.assertTrue(
            cfg["prompts"]["transfer_sort_fruit"]["reference_image"].endswith(
                "transfer/sort_fruit_depth_triptych.png"
            )
        )
        self.assertTrue(
            cfg["prompts"]["transfer_lemon_depth"]["reference_image"].endswith(
                "transfer/legacy_depth_triptych.png"
            )
        )

    def test_sequence_tasks_rejected_for_flashar(self):
        for task in ("interleave_subtask", "interleave_video"):
            with self.assertRaisesRegex(ValueError, "AR eager only"):
                compose_dict(engine="flashar", backend="vllm", task=task)

    def test_interleave_uses_sequence_checkpoint_and_eager_only(self):
        for task in ("interleave_subtask", "interleave_video"):
            for size, suffix in (("4b", "-4B-Sequence"), ("34b", "-Sequence")):
                cfg = compose_dict(engine="ar", backend="eager", task=task, model_size=size, num_samples=1)
                self.assertEqual(cfg["model_path"], "../ckpt/Xiaomi-Robotics-U0" + suffix)
                self.assertEqual(cfg["task_type"], task)
            with self.assertRaisesRegex(ValueError, "eager backend only"):
                compose_dict(engine="ar", backend="vllm", task=task)

    def test_multi_gpu_profile_sets_external_cli_resources(self):
        cases = [
            ("ar", "eager", "t2i", {"device_map": "balanced", "model_device": "auto"}),
            ("flashar", "eager", "x2i", {"device_map": "balanced", "flashar_device": "cuda:0"}),
            ("ar", "vllm", "t2i", {"tensor_parallel_size": 2, "max_num_seqs": 4}),
            ("flashar", "vllm", "transfer", {"tensor_parallel_size": 2, "max_num_seqs": 4}),
        ]
        for engine, backend, task, expected in cases:
            with self.subTest(engine=engine, backend=backend, task=task):
                cfg = compose_dict(
                    engine=engine,
                    backend=backend,
                    task=task,
                    profile="multi-gpu",
                    num_samples=1,
                )
                self.assertEqual(cfg["profile"], "multi_gpu")
                self.assertTrue(cfg["save_path"].startswith("outputs/multigpu/"))
                for key, value in expected.items():
                    self.assertEqual(cfg[key], value)

    def test_multi_gpu_profile_keeps_flashar_prefix_cache_disabled(self):
        cfg = compose_dict(
            engine="flashar",
            backend="vllm",
            task="t2i",
            profile="multi-gpu",
            num_samples=1,
        )
        self.assertIs(cfg["enable_prefix_caching"], False)

    def test_multi_gpu_profile_can_be_overridden_from_cli_layer(self):
        cfg = compose_dict(
            engine="flashar",
            backend="vllm",
            task="t2i",
            profile="multi-gpu",
            num_samples=1,
            overrides={"tensor_parallel_size": 4, "max_num_seqs": 8},
        )
        self.assertEqual(cfg["tensor_parallel_size"], 4)
        self.assertEqual(cfg["max_num_seqs"], 8)


if __name__ == "__main__":
    unittest.main()
