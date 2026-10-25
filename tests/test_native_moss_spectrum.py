"""CPU contract checks against ComfyUI's real native MOSS implementation.

Run with ComfyUI on PYTHONPATH and its torch environment. These tests deliberately
select CPU before importing model_management; no checkpoint or GPU is needed.
"""

import importlib
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

try:
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import comfy.model_sampling
    from comfy.ldm.moss_image.model import MOSSImage
    from comfy.patcher_extension import WrappersMP, WrapperExecutor, get_all_wrappers
    from comfy.utils import deepcopy_list_dict
except ImportError as exc:
    raise unittest.SkipTest(f"Native MOSS CPU tests require a current ComfyUI: {exc}")

torch.set_num_threads(1)
sys.path.insert(0, str(ROOT / "_vendor"))
PACKAGE = "_spectrum_native_cpu_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
moss = importlib.import_module(f"{PACKAGE}.moss")
model_patch_moss = importlib.import_module(f"{PACKAGE}.model_patch_moss")
model_patch = importlib.import_module(f"{PACKAGE}.model_patch")
spectrum = importlib.import_module(f"{PACKAGE}.spectrum")
node_module = importlib.import_module(f"{PACKAGE}.nodes")


class Sampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    pass


def tiny_model(dtype=torch.float32):
    torch.manual_seed(42)
    model = MOSSImage(
        dim=32, ffn_dim=64, num_heads=2, num_layers=2, text_dim=16,
        latent_channels=8, dtype=dtype, device="cpu", operations=comfy.ops.manual_cast,
    ).eval()
    # Comfy ops skip initialization because production weights come from a file.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    return model


class Base:
    def __init__(self, dit=None):
        self.diffusion_model = tiny_model() if dit is None else dit
        self.model_sampling = Sampling(types.SimpleNamespace(
            sampling_settings={"shift": 5.0, "multiplier": 1.0}
        ))
        self.calls = 0
        self.fail = False

    def apply_model(self, x, sigma, c_crossattn, transformer_options=None):
        options = {} if transformer_options is None else transformer_options
        return WrapperExecutor.new_class_executor(
            self._apply_model, self, get_all_wrappers(WrappersMP.APPLY_MODEL, options)
        ).execute(x, sigma, c_crossattn, options)

    def _apply_model(self, x, sigma, c_crossattn, transformer_options):
        self.calls += 1
        out = self.diffusion_model(
            x, self.model_sampling.timestep(sigma).float(), c_crossattn,
            transformer_options={} if transformer_options is None else transformer_options,
        )
        if self.fail:
            raise RuntimeError("injected forward failure")
        return self.model_sampling.calculate_denoised(sigma, out.float(), x)


class Patcher:
    def __init__(self, base=None, options=None):
        self.model = Base() if base is None else base
        self.model_options = {} if options is None else options

    def clone(self):
        return Patcher(self.model, deepcopy_list_dict(self.model_options))

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def get_model_object(self, key):
        import comfy.utils
        return comfy.utils.get_attr(self.model, key)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        from comfy.patcher_extension import add_wrapper_with_key
        add_wrapper_with_key(wrapper_type, key, wrapper, self.model_options, is_model_options=True)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.model_options.get("transformer_options", {}).get("wrappers", {}).get(wrapper_type, {}).pop(key, None)


def state_of(model):
    wrapper = model.model_options["model_function_wrapper"]
    return dict(zip(wrapper.__code__.co_freevars, (
        cell.cell_contents for cell in wrapper.__closure__
    )))["native_scope"].get()


SIGMAS = tuple(1.0 - i * 0.075 for i in range(12)) + (0.0,)


def run_sample(model, sigmas=SIGMAS, body=None, **step_options):
    """Enter the same wrapper boundary as CFGGuider.sample, without GPU setup."""
    def original(noise, latent_image, sampler, actual_sigmas, **kwargs):
        if body is not None:
            return body()
        return [run_step(model, float(sigma), **step_options) for sigma in actual_sigmas[:-1]]
    wrappers = get_all_wrappers(WrappersMP.OUTER_SAMPLE, model.model_options, is_model_options=True)
    return WrapperExecutor.new_executor(original, wrappers).execute(
        None, None, None, torch.tensor(sigmas), latent_shapes=None
    )


def accelerated(model=None, **kwargs):
    return model_patch_moss.apply_moss_model_patch(
        Patcher() if model is None else model,
        steps=12, warmup_steps=3, tail_actual_steps=2, cheby_degree=1,
        history_size=8, window_size=10, flex_window=0, **kwargs,
    )


def run_step(model, sigma, *, branch=0, height=2, width=3, options=None, base=None, value=1.0):
    x = torch.ones(1, 8, height, width) * (branch + value)
    context = torch.ones(1, 3 + branch, 16) * (branch + value)
    transformer_options = deepcopy_list_dict(model.model_options.get("transformer_options", {}))
    transformer_options.update(options or {})
    args = {
        "input": x,
        "timestep": torch.tensor([sigma]),
        "c": {"c_crossattn": context, "transformer_options": transformer_options},
        "cond_or_uncond": [branch],
    }
    with torch.inference_mode():
        return model.model_options["model_function_wrapper"](
            (base or model.model).apply_model, args
        )


class NativeMossSpectrumTest(unittest.TestCase):
    def test_native_head_matches_actual_forward_for_batches_and_dtypes(self):
        for dtype in (torch.float32, torch.bfloat16):
            for batch in (1, 2):
                with self.subTest(dtype=dtype, batch=batch), torch.inference_mode():
                    dit = tiny_model(dtype)
                    features = []
                    handle = dit.layers[-1].register_forward_hook(
                        lambda _module, _args, output: features.append(
                            output[:, :6].detach().clone()
                        )
                    )
                    x = torch.randn(batch, 8, 2, 3, dtype=dtype)
                    time = torch.linspace(0.2, 0.8, batch)
                    context = torch.randn(batch, 5, 16, dtype=dtype)
                    try:
                        expected = dit(x, time, context)
                    finally:
                        handle.remove()
                    actual = model_patch_moss.head(dit, time, features[0], x)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_public_contract_and_disabled_identity(self):
        cls = node_module.SpectrumMossModelPatch
        self.assertIs(node_module.NODE_CLASS_MAPPINGS["SpectrumMossModelPatch"], cls)
        self.assertNotIn("NativeMossSpectrumPatch", node_module.NODE_CLASS_MAPPINGS)
        self.assertFalse(hasattr(cls, "SPECTRUM_CAPABILITIES"))
        self.assertEqual(cls.INPUT_TYPES()["required"]["model"][0], "MODEL")
        self.assertNotIn("spectrum_capabilities", cls.INPUT_TYPES()["required"]["model"][1])
        self.assertEqual(cls.RETURN_TYPES, ("MODEL",))
        self.assertEqual(cls.FUNCTION, "patch")
        self.assertEqual(
            tuple(inspect.signature(cls.patch).parameters),
            tuple(inspect.signature(node_module.DiTSpectrumPatchAdvanced.patch).parameters),
        )
        original = Patcher()
        self.assertIs(cls().patch(original, enabled=False)[0], original)
        self.assertEqual(original.model_options, {})
        patched = cls().patch(original)[0]
        self.assertIs(patched.model, original.model)
        self.assertIs(patched.model.model_sampling, original.model.model_sampling)
        self.assertEqual(patched.model.model_sampling.multiplier, 1.0)
        self.assertEqual(patched.model.model_sampling.shift, 5.0)
        self.assertEqual(original.model_options, {})

    def test_forecast_skips_blocks_and_keeps_cond_uncond_separate(self):
        model = accelerated()
        states = []
        def body():
            for sigma in (1.0, 0.9, 0.8):
                run_step(model, sigma, branch=0)
                run_step(model, sigma, branch=1)
            state = state_of(model)
            states.append(state)
            self.assertEqual(set(state.forecasters), {0, 1})
            self.assertFalse(torch.equal(state.forecasters[0].h[0], state.forecasters[1].h[0]))
            count = model.model.calls
            for branch in (0, 1):
                result = run_step(model, 0.7, branch=branch)
                self.assertEqual(result.shape, (1, 8, 2, 3))
                self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(model.model.calls, count)
            self.assertIsNone(model_patch._NATIVE_CAPTURE.get())
        run_sample(model, body=body)
        self.assertIsNone(state_of(model))
        self.assertEqual(states[0].forecasters, {})

    def test_sigma_restart_shape_change_and_live_model_replacement_rewarm(self):
        model = accelerated()
        def body():
            for sigma in (1.0, 0.9, 0.8, 0.7):
                run_step(model, sigma)
            count = model.model.calls
            run_step(model, 0.6, width=4)
            self.assertEqual(model.model.calls, count + 1)
            self.assertEqual(state_of(model).forecasters[0].shape, (1, 8, 32))
            run_step(model, 1.0)
            self.assertEqual(state_of(model).step_idx, 0)
            self.assertEqual(len(state_of(model).forecasters[0].t), 1)
            replacement = Base()
            run_step(model, 0.9, base=replacement)
            self.assertEqual(replacement.calls, 1)
            self.assertEqual(state_of(model).step_idx, 0)
        run_sample(model, body=body)

    def test_tail_and_cache_veto_require_actual_forwards(self):
        model = accelerated()
        def body():
            for i in range(10):
                run_step(model, 1.0 - i * 0.05)
            count = model.model.calls
            run_step(model, 0.5)
            run_step(model, 0.45)
            self.assertEqual(model.model.calls, count + 2)
        run_sample(model, body=body)
        model = accelerated(Patcher(options={"spectrum_cache_vetoes": [lambda **_: False]}))
        run_sample(model)
        self.assertEqual(model.model.calls, 12)

    def test_downstream_clone_veto_uses_active_patcher_without_affecting_sibling(self):
        original = accelerated()
        clone = original.clone()
        clone.model_options["spectrum_cache_vetoes"] = [lambda **_: False]
        original.model.current_patcher = clone
        run_sample(clone)
        self.assertEqual(original.model.calls, 12)
        original.model.current_patcher = original
        before = original.model.calls
        run_sample(original)
        self.assertLess(original.model.calls - before, 12)
        self.assertNotIn("spectrum_cache_vetoes", original.model_options)

    def test_failure_clears_capture_and_forecast_state(self):
        model = accelerated()
        args = (torch.ones(1, 8, 2, 3), torch.tensor([0.5]), torch.ones(1, 3, 16))
        with torch.inference_mode():
            expected_base = model.model.apply_model(*args)
        states = []
        def body():
            run_step(model, 1.0)
            states.append(state_of(model))
            model.model.fail = True
            run_step(model, 0.9)
        with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
            run_sample(model, body=body)
        self.assertIsNone(state_of(model))
        self.assertEqual(states[0].forecasters, {})
        self.assertIsNone(states[0].features)
        self.assertIsNone(model_patch._NATIVE_CAPTURE.get())
        model.model.fail = False
        with torch.inference_mode():
            actual_base = model.model.apply_model(*args)
        torch.testing.assert_close(actual_base, expected_base, rtol=0, atol=0)
        fresh = accelerated()
        for actual, expected in zip(run_sample(model), run_sample(fresh)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_clones_leave_no_capture_hook_and_keep_forecasts_separate(self):
        base = Patcher()
        args = (torch.ones(1, 8, 2, 3), torch.tensor([0.5]), torch.ones(1, 3, 16))
        with torch.inference_mode():
            expected_base = base.model.apply_model(*args)
        first, second = accelerated(base), accelerated(base)
        def first_body():
            for sigma in (1.0, 0.9, 0.8):
                run_step(first, sigma)
            first_state = state_of(first)
            def second_body():
                run_step(second, 1.0, branch=1)
                self.assertEqual(set(first_state.forecasters), {0})
                self.assertEqual(set(state_of(second).forecasters), {1})
            run_sample(second, body=second_body)
            self.assertIsNone(state_of(second))
            self.assertIs(state_of(first), first_state)
            before = len(first_state.forecasters[0].t)
            with torch.inference_mode():
                actual_base = base.model.apply_model(*args)
            torch.testing.assert_close(actual_base, expected_base, rtol=0, atol=0)
            self.assertEqual(len(first_state.forecasters[0].t), before)
            count = base.model.calls
            run_step(first, 0.7)
            self.assertEqual(base.model.calls, count)
        run_sample(first, body=first_body)
        block = base.model.diffusion_model.layers[-1]
        self.assertEqual(len(block._forward_hooks), 0)
        self.assertFalse(hasattr(block, "_native_spectrum_state"))
        self.assertIsNone(state_of(first))
        self.assertIsNone(model_patch._NATIVE_CAPTURE.get())

    def test_unsafe_wrapper_preserves_calls_and_last_block_replacement_falls_back(self):
        calls = []
        def old_wrapper(apply_model, args):
            calls.append(True)
            return apply_model(args["input"], args["timestep"], **args["c"])
        model = accelerated(Patcher(options={"model_function_wrapper": old_wrapper}))
        run_sample(model)
        self.assertEqual(len(calls), 12)
        model = accelerated()
        options = {"patches_replace": {"dit": {
            ("single_block", 1): lambda args, extra: {
                "img": extra["original_block"](args)["img"] + 0.1
            },
        }}}
        def body():
            for sigma in (1.0, 0.9, 0.8):
                run_step(model, sigma)
            run_step(model, 0.7, options=options)
            self.assertEqual(model.model.calls, 4)
            self.assertEqual(state_of(model).forecasters, {})
        run_sample(model, body=body)

    def test_one_sampler_only_rearms_on_new_execution(self):
        model = accelerated(one_sampler_only=True)
        context = types.SimpleNamespace(prompt_id="first")
        with patch.object(spectrum, "get_executing_context", return_value=context):
            first = run_sample(model)
            first_count = model.model.calls
            self.assertLess(first_count, 12)
            run_sample(model)
            self.assertEqual(model.model.calls, first_count + 12)
            context.prompt_id = "second"
            count = model.model.calls
            restarted = run_sample(model)
            self.assertEqual(model.model.calls - count, first_count)
            for actual, expected in zip(restarted, first):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reused_model_matches_fresh_for_same_and_smaller_crop_schedules(self):
        reused = accelerated()
        variants = [
            (SIGMAS, {}),
            (SIGMAS, {"value": 2.5}),
            (SIGMAS, {"branch": 1, "value": 0.5}),
            # The following run ends above the next run's starting sigma.
            # A sigma-increase heuristic cannot identify that new invocation.
            (tuple(1.0 - i * 0.03 for i in range(12)) + (0.0,), {"value": 2.0}),
            (tuple(x * 0.5 for x in SIGMAS), {"width": 4, "value": 3.0}),
            (tuple(x * 0.5 for x in SIGMAS), {"width": 4, "value": -0.5}),
        ]
        for sigmas, options in variants:
            with self.subTest(sigmas=sigmas, options=options):
                count = reused.model.calls
                actual = run_sample(reused, sigmas, **options)
                fresh = accelerated()
                expected = run_sample(fresh, sigmas, **options)
                self.assertEqual(reused.model.calls - count, fresh.model.calls)
                self.assertLess(fresh.model.calls, len(sigmas) - 1)
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, rtol=0, atol=0)
                self.assertIsNone(state_of(reused))

    def test_repeated_same_single_sigma_and_nested_invocations_get_fresh_state(self):
        model = accelerated()
        states = []
        def one_step():
            state = state_of(model)
            states.append(state)
            self.assertEqual(state.step_idx, -1)
            run_step(model, 0.5)
            self.assertEqual(state.step_idx, 0)
            self.assertEqual(len(state.forecasters[0].t), 1)
        for _ in range(2):
            run_sample(model, (0.5, 0.0), body=one_step)
        def outer():
            original = state_of(model)
            run_step(model, 0.9)
            run_sample(model, (0.5, 0.0), body=one_step)
            self.assertIs(state_of(model), original)
            self.assertEqual(original.step_idx, 0)
            self.assertEqual(len(original.forecasters[0].t), 1)
        run_sample(model, body=outer)
        self.assertEqual(len({id(state) for state in states}), 3)
        self.assertTrue(all(not state.forecasters for state in states))
        self.assertIsNone(state_of(model))

    def test_actual_schedule_length_controls_warmup_and_tail(self):
        model = accelerated()
        for steps in (1, 4, 5, 8):
            sigmas = tuple(0.5 - i * 0.5 / steps for i in range(steps + 1))
            counts = []
            def body():
                self.assertEqual(state_of(model).num_steps, steps)
                for sigma in sigmas[:-1]:
                    before = model.model.calls
                    run_step(model, sigma)
                    counts.append(model.model.calls - before)
            run_sample(model, sigmas, body=body)
            self.assertTrue(all(counts[:min(3, steps)]))
            self.assertTrue(all(counts[-2:]))
            self.assertEqual(sum(counts), min(5, steps))

    def test_unknown_sampling_scope_never_forecasts(self):
        model = accelerated()
        run_sample(model)
        before = model.model.calls
        for sigma in SIGMAS[:-1]:
            actual = run_step(model, sigma, value=3.0)
            with torch.inference_mode():
                expected = model.model.apply_model(
                    torch.ones(1, 8, 2, 3) * 3.0, torch.tensor([sigma]),
                    torch.ones(1, 3, 16) * 3.0,
                )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(model.model.calls - before, 24)
        self.assertIsNone(state_of(model))

    def test_repeated_sigma_observations_do_not_duplicate_taylor_times(self):
        model = accelerated()

        def body():
            for sigma in SIGMAS[:3]:
                run_step(model, sigma, value=0.5)
                run_step(model, sigma, value=1.0)
            self.assertEqual(state_of(model).forecasters[0].t, [0.0, 1.0, 2.0])
            count = model.model.calls
            out = run_step(model, SIGMAS[3])
            self.assertEqual(model.model.calls, count)
            self.assertTrue(torch.isfinite(out).all())
            self.assertLess(out.abs().max().item(), 100.0)

        run_sample(model, body=body)

    def test_scheduled_weight_hooks_clear_forecasts_and_keep_actual_calls(self):
        model = accelerated()

        def body():
            for sigma in SIGMAS[:3]:
                run_step(model, sigma)
            self.assertTrue(state_of(model).forecasters)
            model.model.current_patcher = types.SimpleNamespace(hook_patches={"scheduled": object()})
            before = model.model.calls
            for sigma in SIGMAS[3:-1]:
                run_step(model, sigma)
            self.assertEqual(model.model.calls - before, len(SIGMAS[3:-1]))
            self.assertFalse(state_of(model).forecasters)

        run_sample(model, body=body)

    def test_native_adapter_rejects_custom_loader_and_keeps_legacy_contract(self):
        legacy = Patcher(types.SimpleNamespace(
            diffusion_model=types.SimpleNamespace(**{name: None for name in moss._MOSS_HEAD}),
            model_sampling=object(),
        ))
        self.assertEqual(moss._require_moss(legacy), (legacy.model.diffusion_model, legacy.model.model_sampling))
        with self.assertRaisesRegex(RuntimeError, "native ComfyUI"):
            accelerated(legacy)
        with self.assertRaisesRegex(RuntimeError, "freq_dim"):
            moss._require_moss(Patcher())


if __name__ == "__main__":
    unittest.main()
