from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from configs.compose import compose_dict
from scripts.inference_flashar import render_case
from xr_u0_ar.task_prompts import T2I_PROMPT_TEMPLATE, T2I_UNCOND_PROMPT


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


class T2IPromptConfigTest(unittest.TestCase):
    def test_uncond_keeps_task_context_and_only_clears_user_text(self):
        self.assertEqual(T2I_UNCOND_PROMPT, T2I_PROMPT_TEMPLATE.format(text=""))
        self.assertIn("for t2i task", T2I_UNCOND_PROMPT)

    def test_ar_and_flashar_configs_share_task_specific_uncond(self):
        for engine in ("ar", "flashar"):
            with self.subTest(engine=engine):
                cfg = compose_dict(engine=engine, backend="eager", task="t2i", num_samples=1)
                self.assertEqual(cfg["template"], T2I_PROMPT_TEMPLATE)
                self.assertEqual(cfg["unc_prompt"], T2I_UNCOND_PROMPT)

    def test_unified_flashar_script_renders_task_specific_uncond(self):
        cfg = SimpleNamespace(
            task_type="T2I",
            template=T2I_PROMPT_TEMPLATE,
            unc_prompt=T2I_UNCOND_PROMPT,
        )
        _, uncond = render_case(
            cfg,
            {"id": "case", "prompt": "A ceramic vase."},
            tokenizer=None,
            vision_tokenizer=None,
            height=64,
            width=64,
        )
        self.assertEqual(
            uncond,
            T2I_UNCOND_PROMPT + "<|image start|>64*64<|image token|>",
        )


@unittest.skipUnless(HAS_TORCH and HAS_TRANSFORMERS, "FlashAR entry points require torch and transformers")
class T2IPromptEntryPointTest(unittest.TestCase):
    def test_flashar_package_lazy_export_preserves_public_api(self):
        from xr_u0_flashar import UNISFlashAR
        from xr_u0_flashar.model import UNISFlashAR as ModelClass

        self.assertIs(UNISFlashAR, ModelClass)

    def test_eager_cli_renders_task_specific_uncond(self):
        from xr_u0_flashar.eager_cli import render_t2i_prompts

        _, uncond = render_t2i_prompts("A ceramic vase.", height=64, width=64)
        self.assertEqual(
            uncond,
            T2I_UNCOND_PROMPT + "<|image start|>64*64<|image token|>",
        )

    def test_batch_cli_renders_task_specific_uncond(self):
        from xr_u0_flashar.batch_cli import _render_inputs

        _, unconds = _render_inputs(
            [{"prompt": "A ceramic vase."}],
            tokenizer=None,
            vision_tokenizer=None,
            height=64,
            width=64,
            source_image_area=1024 * 1024,
        )
        self.assertEqual(
            unconds,
            [T2I_UNCOND_PROMPT + "<|image start|>64*64<|image token|>"],
        )

    def test_demo_runtime_renders_task_specific_uncond(self):
        from xr_u0_flashar.demo_runtime import TASK_T2I, render_flashar_request

        rendered = render_flashar_request(
            task_type=TASK_T2I,
            text="A ceramic vase.",
            tokenizer=None,
            vision_tokenizer=None,
        )
        self.assertEqual(
            rendered.uncond_prompt,
            T2I_UNCOND_PROMPT + "<|image start|>64*64<|image token|>",
        )

    def test_vllm_api_default_renders_task_specific_uncond(self):
        from xr_u0_flashar.vllm.api import _DEFAULT_UNCOND_TEMPLATE

        self.assertEqual(
            _DEFAULT_UNCOND_TEMPLATE.format(H=64, W=64),
            T2I_UNCOND_PROMPT + "<|image start|>64*64<|image token|>",
        )


if __name__ == "__main__":
    unittest.main()
