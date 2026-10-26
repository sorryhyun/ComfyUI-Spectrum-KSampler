"""CPU contracts against ComfyUI's real Krea2 implementation; no weights/GPU."""

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
    import comfy.ops
    import comfy.model_sampling
    from comfy.ldm.krea2.model import SingleStreamDiT
    from comfy.patcher_extension import WrappersMP, WrapperExecutor, get_all_wrappers
    from comfy.utils import deepcopy_list_dict
except ImportError as exc:
    raise unittest.SkipTest(f"Native Krea2 CPU tests require a current ComfyUI: {exc}")

torch.set_num_threads(1)
sys.path.insert(0, str(ROOT / "_vendor"))
PACKAGE = "_spectrum_krea2_cpu_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
krea2 = importlib.import_module(f"{PACKAGE}.model_patch_krea2")


class Sampling(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
    pass


def tiny_model(dtype=torch.float32):
    torch.manual_seed(42)
    model = SingleStreamDiT(
        features=64, tdim=16, txtdim=32, heads=2, kvheads=1,
        multiplier=2, layers=2, patch=2, channels=4,
        txtlayers=2, txtheads=2, txtkvheads=1,
        dtype=dtype, device="cpu", operations=comfy.ops.manual_cast,
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    return model


class Base:
    def __init__(self, dit=None):
        self.diffusion_model = tiny_model() if dit is None else dit
        self.model_sampling = Sampling(types.SimpleNamespace(sampling_settings={"shift": 1.15}))
        self.calls = 0
        self.blocks = 0
        self.fail = False
        self.diffusion_model.blocks[0].register_forward_hook(self._count_block)

    def _count_block(self, *_):
        self.blocks += 1

    def apply_model(self, x, sigma, c_crossattn, transformer_options=None, **kwargs):
        options = {} if transformer_options is None else transformer_options
        return WrapperExecutor.new_class_executor(
            self._apply_model, self, get_all_wrappers(WrappersMP.APPLY_MODEL, options)
        ).execute(x, sigma, c_crossattn, options, **kwargs)

    def _apply_model(self, x, sigma, c_crossattn, transformer_options, **kwargs):
        self.calls += 1
        out = self.diffusion_model(
            x, self.model_sampling.timestep(sigma).float(), c_crossattn,
            transformer_options=transformer_options or {}, **kwargs,
        )
        if self.fail:
            raise RuntimeError("injected Krea2 failure")
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
        return comfy.utils.get_attr(self.model, key)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        from comfy.patcher_extension import add_wrapper_with_key
        add_wrapper_with_key(wrapper_type, key, wrapper, self.model_options, is_model_options=True)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.model_options.get("transformer_options", {}).get("wrappers", {}).get(wrapper_type, {}).pop(key, None)


SIGMAS = tuple(1.0 - i / 8 for i in range(9))


def accelerated(model=None, **kwargs):
    settings = dict(window_size=10, flex_window=0)
    settings.update(kwargs)
    return krea2.apply_krea2_model_patch(Patcher() if model is None else model, **settings)


def run_step(model, sigma, *, branch=0, height=5, width=6, temporal=False,
             options=None, conditions=None, base=None, value=1.0, batch=1):
    shape = (batch, 4, 1, height, width) if temporal else (batch, 4, height, width)
    x = torch.ones(shape) * (value + branch)
    context = torch.ones(batch, 3 + branch, 64) * (value + branch)
    opts = deepcopy_list_dict(model.model_options.get("transformer_options", {}))
    opts.update(options or {})
    c = {"c_crossattn": context, "transformer_options": opts}
    c.update(conditions or {})
    args = {"input": x, "timestep": torch.full((batch,), sigma),
            "c": c, "cond_or_uncond": [branch]}
    with torch.inference_mode():
        return model.model_options["model_function_wrapper"](
            (base or model.model).apply_model, args,
        )


def run_sample(model, sigmas=SIGMAS, body=None, **options):
    def original(noise, latent, sampler, actual_sigmas, **kwargs):
        if body is not None:
            return body()
        return [run_step(model, float(sigma), **options) for sigma in actual_sigmas[:-1]]
    wrappers = get_all_wrappers(WrappersMP.OUTER_SAMPLE, model.model_options, is_model_options=True)
    return WrapperExecutor.new_executor(original, wrappers).execute(
        None, None, None, torch.tensor(sigmas), latent_shapes=None,
    )


class NativeKrea2SpectrumTest(unittest.TestCase):
    def test_public_node_contract_matches_api_defaults(self):
        nodes = importlib.import_module(f"{PACKAGE}.nodes")
        cls = nodes.SpectrumKrea2ModelPatch
        self.assertIs(nodes.NODE_CLASS_MAPPINGS["SpectrumKrea2ModelPatch"], cls)
        self.assertNotIn("NativeKrea2SpectrumPatch", nodes.NODE_CLASS_MAPPINGS)
        self.assertFalse(hasattr(cls, "SPECTRUM_CAPABILITIES"))
        inputs = cls.INPUT_TYPES()["required"]
        self.assertEqual(inputs["model"][0], "MODEL")
        self.assertNotIn("spectrum_capabilities", inputs["model"][1])
        self.assertEqual(cls.RETURN_TYPES, ("MODEL",))
        self.assertEqual(cls.FUNCTION, "patch")
        node_params = inspect.signature(cls.patch).parameters
        api_params = inspect.signature(krea2.apply_krea2_model_patch).parameters
        self.assertEqual(tuple(node_params)[1:], tuple(api_params))
        for field, param in api_params.items():
            if field == "model":
                continue
            with self.subTest(field=field):
                self.assertEqual(node_params[field].default, param.default)
                self.assertEqual(inputs[field][1]["default"], param.default)
        original = Patcher()
        self.assertIs(cls().patch(original, enabled=False)[0], original)

    def test_head_matches_native_for_dtypes_batches_temporal_and_odd_padding(self):
        for dtype in (torch.float32, torch.bfloat16):
            for batch in (1, 2):
                for temporal in (False, True):
                    with self.subTest(dtype=dtype, batch=batch, temporal=temporal), torch.inference_mode():
                        dit = tiny_model(dtype)
                        captured = []
                        hook = dit.last.register_forward_pre_hook(
                            lambda _, args: captured.append(args[0].detach().clone()))
                        shape = (batch, 4, 1, 5, 7) if temporal else (batch, 4, 5, 7)
                        x = torch.randn(shape, dtype=dtype)
                        t = torch.linspace(0.2, 0.8, batch)
                        context = torch.randn(batch, 3, 64, dtype=dtype)
                        try:
                            expected = dit(x, t, context)
                        finally:
                            hook.remove()
                        feature = krea2.extract_features(captured[0], x, {"c_crossattn": context})
                        self.assertEqual(feature.shape, (batch, 12, 64))
                        actual = krea2.head(dit, t, feature, x)
                        # Image-only GEMM has fewer rows than the native joint
                        # text/image head and can change FP32 reduction rounding.
                        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=3e-7)

    def test_disabled_identity_sampling_and_original_options_are_preserved(self):
        original = Patcher()
        self.assertIs(krea2.apply_krea2_model_patch(original, enabled=False), original)
        model = krea2.apply_krea2_model_patch(original)
        self.assertIs(model.model, original.model)
        self.assertIs(model.model.model_sampling, original.model.model_sampling)
        self.assertEqual(model.model.model_sampling.shift, 1.15)
        self.assertEqual(original.model_options, {})
        with self.assertRaisesRegex(RuntimeError, "native ComfyUI Krea2"):
            krea2.require_model(types.SimpleNamespace(model=object()))

    def test_forecast_reduces_native_blocks_and_keeps_branches_separate(self):
        model = accelerated()
        def body():
            for sigma in (1.0, 0.875, 0.75):
                run_step(model, sigma, branch=0)
                run_step(model, sigma, branch=1)
            count = model.model.blocks
            cond = run_step(model, 0.625, branch=0)
            uncond = run_step(model, 0.625, branch=1)
            self.assertEqual(model.model.blocks, count)
            self.assertEqual(cond.shape, (1, 4, 5, 6))
            self.assertTrue(torch.isfinite(cond).all())
            self.assertFalse(torch.equal(cond, uncond))
        run_sample(model, body=body)

    def test_default_turbo_schedule_actually_forecasts_and_short_schedules_do_not(self):
        model = krea2.apply_krea2_model_patch(Patcher())
        run_sample(model)
        self.assertLess(model.model.blocks, 8)
        for steps in (1, 3, 5):
            with self.subTest(steps=steps):
                count = model.model.blocks
                run_sample(model, tuple(1 - i / steps for i in range(steps + 1)))
                self.assertEqual(model.model.blocks - count, steps)

    def test_shared_runner_keeps_replacement_context_keys_separate(self):
        # Current native Krea2 does not dispatch patches_replace. This verifies
        # shared-runner identity and lifetime, not Krea2 SLG support.
        model = accelerated(compat_policy="strict")
        ordinary = {"uuids": ["same-conditioning"]}
        replacement = dict(ordinary, patches_replace={"dit": {("single_block", 0): lambda a, e: a}})
        states = []

        def body():
            wrapper = model.model_options["model_function_wrapper"]
            state = inspect.getclosurevars(wrapper).nonlocals["native_scope"].get()
            states.append(state)
            for sigma in (1.0, 0.875, 0.75):
                run_step(model, sigma, options=ordinary)
                run_step(model, sigma, options=replacement)
            self.assertEqual(len(state.forecasters), 2)
            self.assertEqual(model.model.blocks, 6)
            run_step(model, 0.625, options=ordinary)
            run_step(model, 0.625, options=replacement)
            self.assertEqual(model.model.blocks, 6)

        run_sample(model, body=body)
        self.assertFalse(states[0].native_patch_callables)

    def test_repeated_invocations_match_fresh_with_changed_crop_branch_and_temporal(self):
        reused = accelerated()
        for options in ({}, {"value": 2.0}, {"branch": 1}, {"width": 7}, {"temporal": True, "batch": 2}):
            with self.subTest(options=options):
                before = reused.model.blocks
                actual = run_sample(reused, **options)
                fresh = accelerated()
                expected = run_sample(fresh, **options)
                self.assertEqual(reused.model.blocks - before, fresh.model.blocks)
                self.assertLess(fresh.model.blocks, 8)
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, rtol=0, atol=0)

    def test_raw_style_long_schedule_keeps_separate_and_batched_cfg_branches(self):
        # RAW uses the same architecture with a longer downstream schedule.
        # The ignored steps=8 input must not truncate this 52-step scope.
        sigmas = tuple(1 - i / 52 for i in range(53))

        def sample(model, batched):
            outputs = []
            counts = []

            def body():
                for step_index, sigma in enumerate(sigmas[:-1]):
                    before = model.model.blocks
                    if batched:
                        x = torch.cat([torch.ones(1, 4, 5, 6), torch.full((1, 4, 5, 6), 2.0)])
                        context = torch.cat([torch.ones(1, 3, 64), torch.full((1, 3, 64), -0.75)])
                        with torch.inference_mode():
                            result = model.model_options["model_function_wrapper"](
                                model.model.apply_model,
                                {"input": x, "timestep": torch.full((2,), sigma),
                                 "c": {"c_crossattn": context, "transformer_options":
                                       deepcopy_list_dict(model.model_options.get("transformer_options", {}))},
                                 "cond_or_uncond": [0, 1]},
                            )
                    else:
                        result = torch.cat([run_step(model, sigma, branch=0),
                                            run_step(model, sigma, branch=1)])
                    outputs.append(result)
                    counts.append(model.model.blocks - before)
                    if step_index == 2:
                        wrapper = model.model_options["model_function_wrapper"]
                        state = inspect.getclosurevars(wrapper).nonlocals["native_scope"].get()
                        self.assertEqual(set(state.forecasters), {0, 1})
                        self.assertFalse(torch.equal(state.forecasters[0].h[-1],
                                                     state.forecasters[1].h[-1]))

            run_sample(model, sigmas=sigmas, body=body)
            full_calls = 1 if batched else 2
            self.assertEqual(len(outputs), 52)
            self.assertLess(sum(counts), 52 * full_calls)
            self.assertEqual(counts[-2:], [full_calls, full_calls])
            self.assertTrue(all(torch.isfinite(output).all() for output in outputs))
            self.assertFalse(torch.equal(outputs[-1][0], outputs[-1][1]))
            return outputs

        for batched in (False, True):
            with self.subTest(batched=batched):
                reused = accelerated(steps=8)
                first = sample(reused, batched)
                repeated = sample(reused, batched)
                fresh = sample(accelerated(steps=8), batched)
                for a, b, c in zip(first, repeated, fresh):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_exception_cleanup_and_shared_base_are_neutral(self):
        base = Patcher()
        first, second = accelerated(base), accelerated(base)
        args = (torch.ones(1, 4, 5, 6), torch.tensor([0.5]), torch.ones(1, 3, 64))
        with torch.inference_mode():
            expected = base.model.apply_model(*args)
        def failed():
            run_step(first, 1.0)
            base.model.fail = True
            run_step(first, 0.875)
        with self.assertRaisesRegex(RuntimeError, "injected Krea2 failure"):
            run_sample(first, body=failed)
        base.model.fail = False
        with torch.inference_mode():
            actual = base.model.apply_model(*args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        one, two = run_sample(first), run_sample(second)
        for a, e in zip(one, two):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        self.assertEqual(len(base.model.diffusion_model.last._forward_pre_hooks), 0)

    def test_nested_clones_keep_the_outer_history(self):
        base = Patcher()
        first, second = accelerated(base), accelerated(base)
        def body():
            for sigma in (1.0, 0.875, 0.75):
                run_step(first, sigma)
            run_sample(second, value=2.0)
            count = base.model.blocks
            run_step(first, 0.625)
            self.assertEqual(base.model.blocks, count)
        run_sample(first, body=body)

    def test_unknown_scope_and_unsafe_existing_wrapper_use_actual(self):
        model = accelerated()
        run_sample(model)
        before = model.model.blocks
        for sigma in SIGMAS[:-1]:
            run_step(model, sigma)
        self.assertEqual(model.model.blocks - before, 8)
        calls = []
        def wrapper(apply_model, args):
            calls.append(True)
            return apply_model(args["input"], args["timestep"], **args["c"])
        model = accelerated(Patcher(options={"model_function_wrapper": wrapper}))
        run_sample(model)
        self.assertEqual(len(calls), 8)
        self.assertEqual(model.model.blocks, 8)

    def test_reference_and_post_input_paths_use_actual(self):
        for kwargs in (
            {"conditions": {"ref_latents": [torch.zeros(1, 4, 4, 4)], "ref_latents_method": "index"}},
            {"options": {"patches": {"post_input": [lambda args: args]}}},
        ):
            with self.subTest(kwargs=kwargs):
                model = accelerated()
                run_sample(model, **kwargs)
                self.assertEqual(model.model.blocks, 8)
        dit = tiny_model()
        self.assertFalse(krea2.capture_safe(dit, torch.ones(1, 4, 2, 5, 6), {}))

    def test_activation_quantized_head_uses_actual_but_weight_only_head_is_safe(self):
        model = accelerated()
        linear = model.model.diffusion_model.last.linear
        linear.layout_type = "TensorCoreFP8Layout"
        linear.quant_format = "float8_e4m3fn"
        # Simulate the capability flags on native manual_cast ops; no quant GPU kernel needed.
        run_sample(model)
        self.assertEqual(model.model.blocks, 8)
        linear.quant_format = "int8_tensorwise"
        self.assertTrue(krea2.capture_safe(model.model.diffusion_model, torch.ones(1, 4, 5, 6), {}))
        linear.quant_format = "float8_e4m3fn"
        linear.weight_function = [lambda weight: weight]
        self.assertTrue(krea2.capture_safe(model.model.diffusion_model, torch.ones(1, 4, 5, 6), {}))

    def test_apply_and_diffusion_wrappers_allow_forecasts_without_registration(self):
        for kind in (WrappersMP.APPLY_MODEL, WrappersMP.DIFFUSION_MODEL):
            with self.subTest(kind=kind):
                model = Patcher()
                calls = []
                def wrapper(executor, *args, **kwargs):
                    calls.append(True)
                    return executor(*args, **kwargs)
                model.add_wrapper_with_key(kind, "test", wrapper)
                model = accelerated(model)
                run_sample(model)
                self.assertLess(model.model.blocks, 8)
                self.assertEqual(len(calls), model.model.blocks)

    def test_real_core_compile_keeps_live_capture_and_restores_modules(self):
        from comfy_api.torch_helpers import set_torch_compile_wrapper

        model = Patcher()
        original_blocks = tuple(model.model.diffusion_model.blocks)
        original_last = model.model.diffusion_model.last
        set_torch_compile_wrapper(
            model, backend="eager", dynamic=True,
            keys=["diffusion_model.blocks.0", "diffusion_model.blocks.1", "diffusion_model.last"],
        )
        model = accelerated(model)
        seen = []
        original_capture = krea2.capture
        def capture(dit):
            seen.append(type(dit.last).__name__)
            return original_capture(dit)
        with patch.object(krea2, "capture", side_effect=capture):
            result = run_sample(model)
        self.assertLess(model.model.blocks, 8)
        self.assertTrue(seen and all(name == "OptimizedModule" for name in seen))
        self.assertTrue(all(torch.isfinite(value).all() for value in result))
        self.assertTrue(all(actual is original for actual, original in
                            zip(model.model.diffusion_model.blocks, original_blocks)))
        self.assertIs(model.model.diffusion_model.last, original_last)
        self.assertFalse(original_last._forward_pre_hooks)

    def test_control_condition_uses_actual(self):
        model = accelerated()
        run_sample(model, conditions={"control": object()})
        self.assertEqual(model.model.blocks, 8)

    def test_scheduled_weight_hooks_use_actual(self):
        for attr in ("hook_patches", "current_hooks"):
            with self.subTest(attr=attr):
                model = accelerated()
                model.model.current_patcher = types.SimpleNamespace(**{attr: object()})
                run_sample(model)
                self.assertEqual(model.model.blocks, 8)

    def test_live_model_replacement_and_crop_change_require_actual(self):
        model = accelerated()
        def body():
            for sigma in (1.0, 0.875, 0.75, 0.625):
                run_step(model, sigma)
            before = model.model.blocks
            run_step(model, 0.5, width=8)
            self.assertEqual(model.model.blocks, before + 1)
            replacement = Base()
            run_step(model, 0.375, base=replacement)
            self.assertEqual(replacement.blocks, 1)
        run_sample(model, body=body)


if __name__ == "__main__":
    unittest.main()
