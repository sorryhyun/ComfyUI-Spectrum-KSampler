"""CPU regression for the custom moss-image loader's existing forecast path.

The toy model follows its per-sample RMSNorm capture and raw-time head contract;
it does not depend on either ComfyUI's native MOSS implementation or a checkpoint.
Run with ComfyUI on PYTHONPATH and its torch environment.
"""

import builtins
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    from torch import nn
    from comfy.utils import deepcopy_list_dict
except ImportError as exc:
    raise unittest.SkipTest(f"MOSS forecast CPU tests require ComfyUI and torch: {exc}")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_vendor"))
PACKAGE = "_spectrum_custom_moss_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
moss = importlib.import_module(f"{PACKAGE}.moss")
feature_forecast = importlib.import_module(f"{PACKAGE}.feature_forecast")


class CustomLoaderToy(nn.Module):
    """Small custom-loader shape: full forward visits final_norm once per image."""

    def __init__(self):
        super().__init__()
        self.freq_dim = 8
        with torch.random.fork_rng():
            torch.manual_seed(2718)
            self.input_projection = nn.Linear(4, 8)
            self.layers = nn.ModuleList([nn.Linear(8, 8)])
            self.time_embedding = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8))
            self.video_head_modulation = nn.Linear(8, 16)
            self.final_norm_video = nn.RMSNorm(8, eps=1e-6)
            self.final_linear_video = nn.Linear(8, 4, bias=False)
        self.forward_calls = 0
        self.sample_calls = 0

    def reference_head(self, features, raw_time, height, width):
        # The custom loader receives sigma * 1000 already. Applying the native
        # timestep embedding's additional 1000 factor would change this result.
        frequencies = torch.tensor([
            10000.0 ** (-i / (self.freq_dim // 2))
            for i in range(self.freq_dim // 2)
        ], dtype=torch.float64)
        phase = raw_time.double().reshape(-1, 1) * frequencies
        embedding = torch.cat([torch.cos(phase), torch.sin(phase)], dim=1).float()
        modulation = self.video_head_modulation(self.time_embedding(embedding))
        shift, scale = modulation.chunk(2, dim=-1)
        normalized = self.final_norm_video(features)
        output = self.final_linear_video(normalized * (1 + scale[:, None]) + shift[:, None])
        return output.transpose(1, 2).reshape(features.shape[0], 4, height, width)

    def forward(self, x, timestep, context):
        self.forward_calls += 1
        outputs = []
        for image, time, conditioning in zip(x, timestep, context):
            self.sample_calls += 1
            tokens = image.flatten(1).transpose(0, 1).unsqueeze(0)
            features = self.input_projection(tokens) + conditioning.mean() + time / 1000
            for block in self.layers:
                features = torch.tanh(block(features))
            outputs.append(self.reference_head(features, time, *x.shape[-2:]))
        return torch.cat(outputs)


class FlowSampling:
    multiplier = 1000.0
    shift = 5.0

    def timestep(self, sigma):
        return sigma * self.multiplier

    def calculate_denoised(self, sigma, prediction, x):
        return x - prediction * sigma.reshape(-1, 1, 1, 1)


class ToyBase:
    def __init__(self):
        self.diffusion_model = CustomLoaderToy().eval()
        self.model_sampling = FlowSampling()

    def apply_model(self, x, sigma, c_crossattn, transformer_options=None):
        prediction = self.diffusion_model(x, self.model_sampling.timestep(sigma), c_crossattn)
        return self.model_sampling.calculate_denoised(sigma, prediction, x)


class ToyPatcher:
    def __init__(self, model=None, options=None):
        self.model = ToyBase() if model is None else model
        self.model_options = {} if options is None else options

    def clone(self):
        return ToyPatcher(self.model, deepcopy_list_dict(self.model_options))

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def inputs(sigma):
    x = torch.arange(48, dtype=torch.float32).reshape(2, 4, 2, 3) / 48
    context = torch.stack([torch.ones(3, 4), torch.full((3, 4), -0.75)])
    return x, torch.full((2,), sigma), context


def run_step(model, sigma):
    x, timestep, context = inputs(sigma)
    return model.model_options["model_function_wrapper"](model.model.apply_model, {
        "input": x, "timestep": timestep,
        "c": {"c_crossattn": context, "transformer_options": {}},
        "cond_or_uncond": [0, 1],
    })


class MossForecastReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_custom_head_matches_per_sample_reference(self):
        dit = CustomLoaderToy().eval()
        features = torch.linspace(-1, 1, 96).reshape(2, 6, 8)
        times = torch.tensor([750.0, 250.0])
        with torch.inference_mode():
            expected = torch.cat([
                dit.reference_head(features[i:i + 1], times[i:i + 1], 2, 3)
                for i in range(2)
            ])
            actual = moss._moss_head(dit, times, features, 2, 3)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_actual_and_cached_custom_loader_path_preserves_branch_head(self):
        original = ToyPatcher()
        model = moss.apply_moss_spectrum_patch(
            original, steps=8, warmup_steps=3, tail_actual_steps=2,
            cheby_degree=1, history_size=8, window_size=10, flex_window=0,
            compile=False,
        )
        dit = model.model.diffusion_model
        self.assertIs(model.model, original.model)
        self.assertIs(model.model.model_sampling, original.model.model_sampling)
        self.assertEqual(original.model_options, {})
        with torch.inference_mode():
            for sigma in (1.0, 0.9, 0.8):
                actual = run_step(model, sigma)
                baseline = ToyBase().apply_model(*inputs(sigma))
                torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
            state = dit.final_norm_video._spectrum_state
            self.assertEqual(set(state.forecasters), {0, 1})
            self.assertIsInstance(state, moss.MossSpectrumState)
            self.assertTrue(all(isinstance(f, feature_forecast._Forecaster) for f in state.forecasters.values()))
            self.assertFalse(torch.equal(state.forecasters[0].h[0], state.forecasters[1].h[0]))
            self.assertEqual((dit.forward_calls, dit.sample_calls), (3, 6))
            features = torch.cat([state.forecasters[key].predict(3.0) for key in (0, 1)])
            x, sigma, _ = inputs(0.7)
            expected_noise = torch.cat([
                dit.reference_head(features[i:i + 1], sigma[i:i + 1] * 1000, 2, 3)
                for i in range(2)
            ])
            expected = model.model.model_sampling.calculate_denoised(sigma, expected_noise, x)
            actual = run_step(model, 0.7)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            self.assertEqual((dit.forward_calls, dit.sample_calls), (3, 6))
            for sigma in (0.6, 0.5, 0.4, 0.3):
                self.assertTrue(torch.isfinite(run_step(model, sigma)).all())
            # Eight sampler steps: three warmup + two tail full forwards.
            self.assertEqual((dit.forward_calls, dit.sample_calls), (5, 10))
            self.assertIsNone(state.moss_feats)

    def test_same_step_correction_replaces_observation_before_taylor_prediction(self):
        forecast = feature_forecast._Forecaster(m=1, lam=0.1, w=0.0, total_steps=8, k=8)
        for step in range(3):
            forecast.update(step, torch.full((1, 2, 4), float(step - 10)))
            forecast.update(step, torch.full((1, 2, 4), float(step)))
        self.assertEqual(forecast.t, [0.0, 1.0, 2.0])
        torch.testing.assert_close(forecast.predict(3), torch.full((1, 2, 4), 3.0), rtol=0, atol=0)

    def test_disabled_custom_patch_is_identity_without_compile(self):
        model = ToyPatcher()
        with patch.object(moss, "_cpp_compiler_available", side_effect=AssertionError("compile probe")):
            self.assertIs(moss.apply_moss_spectrum_patch(model, enabled=False, compile=False), model)
        self.assertEqual(model.model_options, {})
        self.assertEqual(len(model.model.diffusion_model.final_norm_video._forward_pre_hooks), 0)

    def test_custom_module_import_does_not_require_native_architectures(self):
        real_import = builtins.__import__
        def without_native(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("comfy.ldm.moss_image"):
                raise ModuleNotFoundError(f"Simulated older ComfyUI without {name}")
            if name == "comfy.ldm" and any(str(item).startswith("moss_image") for item in fromlist):
                raise ModuleNotFoundError("Simulated older ComfyUI without native architecture")
            return real_import(name, globals, locals, fromlist, level)
        # A fresh package namespace re-executes this pack's modules; ComfyUI's
        # common sampler machinery stays loaded. No subprocess or GPU is used.
        fresh_name = PACKAGE + "_without_native"
        fresh = types.ModuleType(fresh_name)
        fresh.__path__ = [str(ROOT)]
        sys.modules[fresh_name] = fresh
        try:
            with patch("builtins.__import__", side_effect=without_native):
                imported = importlib.import_module(f"{fresh_name}.moss")
                self.assertTrue(callable(imported.apply_moss_spectrum_patch))
                original = ToyPatcher()
                self.assertIs(imported.apply_moss_spectrum_patch(original, enabled=False, compile=False), original)
        finally:
            for name in list(sys.modules):
                if name == fresh_name or name.startswith(fresh_name + "."):
                    del sys.modules[name]


if __name__ == "__main__":
    unittest.main()
