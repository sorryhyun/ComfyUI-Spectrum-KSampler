import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCALES = ("ko", "ja", "zh", "zh-CN")

EXPECTED_INPUTS = {
    "SpectrumKSampler": {
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise",
        "quality_tags",
        "quality_neg",
        "mod_w_profile",
        "refresh_ratio",
        "adaptive_smc_alpha",
        "fsg",
        "clip",
    },
    "SpectrumKSamplerAdvanced": {
        "model",
        "clip",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise",
        "adapter",
        "quality_tags",
        "quality_neg",
        "mod_w",
        "mod_start_layer",
        "mod_end_layer",
        "mod_taper",
        "mod_taper_scale",
        "mod_final_w",
        "window_size",
        "flex_window",
        "warmup_steps",
        "blend_w",
        "cheby_degree",
        "ridge_lambda",
        "dcw_mode",
        "dcw_lambda",
        "dcw_band_mask",
        "dcw_calibrator",
        "cfgpp_lambda",
        "fsg",
        "fsg_band_lo",
        "fsg_band_hi",
        "fsg_k",
        "fsg_d_sigma",
        "fsg_gamma",
        "adaptive_smc_alpha",
        "smc_cfg_lambda",
    },
    "SpectrumSPDKSampler": {
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise",
        "split_mode",
        "spd_scale",
        "spd_sigma",
        "adaptive_smc_alpha",
    },
    "SpectrumSPDLoRAKSampler": {
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise",
        "lora_name",
        "lora_strength",
        "adaptive_smc_alpha",
    },
    "AnimaModGuidance": {
        "model",
        "clip",
        "quality_tags",
        "quality_neg",
        "mod_w_profile",
        "positive",
        "negative",
    },
    "DiTCFGFSGPatch": {
        "model",
        "enabled",
        "dcw_mode",
        "dcw_lambda",
        "dcw_band_mask",
        "dcw_calibrator",
        "smc_cfg",
        "adaptive_smc_alpha",
        "smc_cfg_lambda",
        "cfgpp",
        "cfgpp_lambda",
        "fsg",
        "fsg_band_lo",
        "fsg_band_hi",
        "fsg_k",
        "fsg_d_sigma",
        "fsg_gamma",
        "replace_existing_cfg",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
        "clip",
        "positive",
    },
    "DiTSpectrumPatch": {
        "model",
        "steps",
        "window_size",
        "flex_window",
        "warmup_steps",
        "tail_actual_steps",
        "blend_w",
        "cheby_degree",
        "ridge_lambda",
        "history_size",
        "enabled",
        "one_sampler_only",
        "verbose",
    },
    "DiTSpectrumPatchAdvanced": {
        "model",
        "steps",
        "window_size",
        "flex_window",
        "warmup_steps",
        "tail_actual_steps",
        "blend_w",
        "cheby_degree",
        "ridge_lambda",
        "history_size",
        "enabled",
        "one_sampler_only",
        "verbose",
        "compat_policy",
    },
}


def _load(locale):
    path = ROOT / "locales" / locale / "nodeDefs.json"
    return json.loads(path.read_text(encoding="utf-8"))


class LocaleNodeDefsTest(unittest.TestCase):
    def test_locales_cover_displayed_nodes_without_socket_renames(self):
        for locale in LOCALES:
            with self.subTest(locale=locale):
                data = _load(locale)
                self.assertEqual(set(data), set(EXPECTED_INPUTS))
                for node_id, inputs in EXPECTED_INPUTS.items():
                    node = data[node_id]
                    self.assertLessEqual(set(node), {"display_name", "description", "inputs"})
                    self.assertIn("display_name", node)
                    self.assertTrue(node["display_name"])
                    self.assertIn("description", node)
                    self.assertEqual(set(node["inputs"]), inputs)
                    for entry in node["inputs"].values():
                        self.assertEqual(set(entry), {"tooltip"})
                        self.assertTrue(entry["tooltip"])


if __name__ == "__main__":
    unittest.main()
