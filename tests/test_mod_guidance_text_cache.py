import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    module_name = "spectrum_mod_guidance_text_cache_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "mod_guidance_text_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeCLIP:
    def __init__(self):
        self.cond_stage_model = torch.nn.Identity()
        self.patcher = SimpleNamespace(patches_uuid="base")
        self.layer_idx = None
        self.tokenizer_options = {}
        self.encodes = []

    def tokenize(self, text):
        return text

    def encode_from_tokens(self, tokens, return_pooled, return_dict):
        self.encodes.append(tokens)
        return {
            "cond": torch.full((1, 2, 3), float(len(self.encodes))),
            "t5xxl_ids": torch.tensor([1, 2]),
            "t5xxl_weights": torch.tensor([1.0, 1.0]),
        }


class EncodeTextCachedTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()
        self.encode = self.module._encode_text_cached

    def test_second_run_reuses_the_first_encode(self):
        clip = FakeCLIP()
        first = self.encode(clip, "quality")
        second = self.encode(clip, "quality")
        self.assertEqual(clip.encodes, ["quality"])
        self.assertIs(first[0], second[0])
        self.assertEqual(first[0].shape, (2, 3))

    def test_other_text_encodes(self):
        clip = FakeCLIP()
        self.encode(clip, "tags")
        self.encode(clip, "neg")
        self.assertEqual(clip.encodes, ["tags", "neg"])

    def test_te_state_changes_encode_again(self):
        clip = FakeCLIP()
        self.encode(clip, "quality")
        clip.patcher.patches_uuid = "lora"
        self.encode(clip, "quality")
        clip.layer_idx = -2
        self.encode(clip, "quality")
        clip.tokenizer_options = {"max": 77}
        self.encode(clip, "quality")
        self.assertEqual(len(clip.encodes), 4)

    def test_missing_t5_keys_stay_none(self):
        clip = FakeCLIP()
        clip.encode_from_tokens = lambda *a, **k: {"cond": torch.zeros(1, 2, 3)}
        self.assertEqual(self.encode(clip, "quality")[1:], (None, None))

    def test_oldest_text_is_evicted(self):
        clip = FakeCLIP()
        size = self.module._TEXT_CACHE_SIZE
        for i in range(size + 1):
            self.encode(clip, str(i))
        self.encode(clip, "0")
        self.assertEqual(len(clip.encodes), size + 2)
        self.assertEqual(len(self.module._TEXT_CACHE[clip.cond_stage_model]), size)


if __name__ == "__main__":
    unittest.main()
