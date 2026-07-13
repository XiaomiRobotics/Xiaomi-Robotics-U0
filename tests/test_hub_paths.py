from __future__ import annotations

import os
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from xr_u0_ar.hub_paths import (
    hub_kwargs_from_config,
    resolve_hub_or_local,
    resolve_local_file,
    resolve_model_path,
)


class HubPathResolverTest(unittest.TestCase):
    def test_existing_local_path_does_not_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch("xr_u0_ar.hub_paths._snapshot_download") as download:
            resolved = resolve_model_path(tmp)
            self.assertEqual(resolved, str(Path(tmp).resolve()))
            download.assert_not_called()

    def test_hub_id_downloads_to_cache(self):
        with tempfile.TemporaryDirectory() as tmp, patch("xr_u0_ar.hub_paths._snapshot_download") as download:
            download.return_value = tmp
            resolved = resolve_hub_or_local("org/xr-u0", kind="model", revision="main", cache_dir="/cache")
            self.assertEqual(resolved, tmp)
            download.assert_called_once_with(
                repo_id="org/xr-u0",
                local_files_only=False,
                revision="main",
                cache_dir="/cache",
            )

    def test_env_local_files_only_is_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"XR_U0_HF_LOCAL_FILES_ONLY": "1"}):
            with patch("xr_u0_ar.hub_paths._snapshot_download") as download:
                download.return_value = tmp
                resolve_model_path("org/xr-u0")
            self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_missing_pathlike_value_is_not_treated_as_repo_id(self):
        with self.assertRaises(FileNotFoundError):
            resolve_model_path("/definitely/missing/xr-u0")

    def test_config_kwargs_from_namespace_and_dict(self):
        ns = SimpleNamespace(hf_revision="v1", hf_cache_dir="/cache", hf_local_files_only=True)
        self.assertEqual(
            hub_kwargs_from_config(ns),
            {"revision": "v1", "cache_dir": "/cache", "local_files_only": True},
        )
        self.assertEqual(
            hub_kwargs_from_config({"hf_revision": "v2"}),
            {"revision": "v2", "cache_dir": None, "local_files_only": None},
        )

    def test_resolve_local_file_prefers_model_dir_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "model.safetensors.index.json"
            state.write_text("{}", encoding="utf-8")
            resolved = resolve_local_file("model.safetensors.index.json", kind="state", base_dir=root)
            self.assertEqual(resolved, str(state.resolve()))


class TokenizerResolutionTest(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("transformers") is None, "transformers is not installed")
    def test_text_tokenizer_uses_resolved_special_tokens_file(self):
        from xr_u0_ar.utils import build_text_tokenizer

        with tempfile.TemporaryDirectory() as tmp, patch("xr_u0_ar.hub_paths._snapshot_download") as download:
            root = Path(tmp)
            (root / "unis_vision_tokens.txt").write_text("<|visual token 000000|>\n", encoding="utf-8")
            download.return_value = tmp
            fake_tokenizer = SimpleNamespace()
            with patch("xr_u0_ar.utils.AutoTokenizer.from_pretrained", return_value=fake_tokenizer) as from_pretrained:
                tokenizer = build_text_tokenizer("org/tokenizer")

        self.assertIs(tokenizer, fake_tokenizer)
        from_pretrained.assert_called_once()
        _, kwargs = from_pretrained.call_args
        self.assertEqual(kwargs["special_tokens_file"], str(root / "unis_vision_tokens.txt"))


class VisionTokenizerResolutionTest(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("omegaconf") is None, "omegaconf is not installed")
    def test_legacy_vision_tokenizer_uses_config_and_ckpt_after_download(self):
        from xr_u0_ar.vision_tokenizer import build_vision_tokenizer

        class FakeIBQ:
            def load_state_dict(self, state):
                self.state = state

            def eval(self):
                return self

            def to(self, device):
                self.device = device
                return self

        with tempfile.TemporaryDirectory() as tmp, patch("xr_u0_ar.hub_paths._snapshot_download") as download:
            root = Path(tmp)
            (root / "config.yaml").write_text("{}", encoding="utf-8")
            (root / "model.ckpt").write_bytes(b"placeholder")
            download.return_value = tmp
            with patch("xr_u0_ar.vision_tokenizer.OmegaConf.load", return_value={}), \
                    patch("xr_u0_ar.vision_tokenizer.torch.load", return_value={"state_dict": {"w": 1}}), \
                    patch("xr_u0_ar.vision_tokenizer.IBQ", return_value=FakeIBQ()) as ibq:
                tokenizer = build_vision_tokenizer("ibq", "org/vision-tokenizer", device="cpu")

        ibq.assert_called_once_with()
        self.assertEqual(tokenizer.state, {"w": 1})
        self.assertEqual(tokenizer.device, "cpu")


class FlashARStateDiscoveryTest(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("transformers") is None, "transformers is not installed")
    def test_flashar_state_is_discovered_from_model_dir(self):
        from scripts.inference_flashar import discover_flashar_state

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "model.safetensors.index.json"
            state.write_text('{"weight_map": {}}', encoding="utf-8")
            cfg = SimpleNamespace(model_path=tmp)
            self.assertEqual(discover_flashar_state(cfg, model_dir=tmp), str(state.resolve()))


if __name__ == "__main__":
    unittest.main()
