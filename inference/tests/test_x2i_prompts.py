from __future__ import annotations

import unittest

from configs.compose import compose_dict
from scripts.inference_flashar import DEFAULT_X2I_TEMPLATE, DEFAULT_X2I_UNCOND
from xr_u0_ar.task_prompts import X2I_PROMPT_TEMPLATE, X2I_UNCOND_PROMPT


EXPECTED_X2I_PROMPT_TEMPLATE = (
    "<|extra_203|>You are a helpful assistant. "
    "USER: <|IMAGE|>{question} ASSISTANT: <|extra_100|>"
)


class X2IPromptConfigTest(unittest.TestCase):
    def test_cond_prompt_uses_generic_assistant_context(self):
        self.assertEqual(X2I_PROMPT_TEMPLATE, EXPECTED_X2I_PROMPT_TEMPLATE)
        self.assertNotIn("for x2i task", X2I_PROMPT_TEMPLATE)

    def test_uncond_only_clears_user_instruction(self):
        self.assertEqual(X2I_UNCOND_PROMPT, X2I_PROMPT_TEMPLATE.format(question=""))

    def test_ar_and_flashar_configs_share_x2i_prompts(self):
        for engine in ("ar", "flashar"):
            with self.subTest(engine=engine):
                cfg = compose_dict(engine=engine, backend="eager", task="x2i", num_samples=1)
                self.assertEqual(cfg["template"], X2I_PROMPT_TEMPLATE)
                self.assertEqual(cfg["unc_prompt"], X2I_UNCOND_PROMPT)

    def test_unified_flashar_defaults_share_x2i_prompts(self):
        self.assertEqual(DEFAULT_X2I_TEMPLATE, X2I_PROMPT_TEMPLATE)
        self.assertEqual(DEFAULT_X2I_UNCOND, X2I_UNCOND_PROMPT)


if __name__ == "__main__":
    unittest.main()
