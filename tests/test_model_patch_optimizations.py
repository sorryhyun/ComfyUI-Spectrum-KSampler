"""Native model patches compose with ordinary optimization wrappers on CPU."""
import unittest
from unittest.mock import patch

from test_native_moss_spectrum import (
    Patcher, WrappersMP, accelerated, model_patch, model_patch_moss,
    run_sample, run_step, state_of, torch,
)


def feature_wrapper(executor, *args, **kwargs):
    """Same lifecycle as DAVE: bind a live block, modify features, then unhook."""
    dit = getattr(executor.class_obj, "diffusion_model", executor.class_obj)
    block = dit.layers[-1]
    handle = block.register_forward_hook(lambda module, values, output: output + 0.25)
    try:
        return executor(*args, **kwargs)
    finally:
        handle.remove()


class ModelPatchOptimizationsTest(unittest.TestCase):
    def test_apply_and_diffusion_wrappers_need_no_registration(self):
        for wrapper_type in (WrappersMP.APPLY_MODEL, WrappersMP.DIFFUSION_MODEL):
            with self.subTest(wrapper_type=wrapper_type):
                model = Patcher()
                model.add_wrapper_with_key(wrapper_type, "optimization",
                                           lambda executor, *a, **k: executor(*a, **k))
                model = accelerated(model, verbose=True)
                with self.assertLogs(model_patch.logger, level="INFO") as logs:
                    run_sample(model)
                self.assertEqual(model.model.calls, 5)
                self.assertTrue(any("sampling ended actual=5 forecast=7" in line
                                    for line in logs.output))

    def test_live_capture_observes_post_hook_features_and_cleans_up(self):
        model = Patcher()
        model.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "feature-modifier", feature_wrapper)
        model = accelerated(model)
        def body():
            output = run_step(model, 1.0)
            feature = state_of(model).forecasters[0].h[-1]
            x, t = torch.ones(1, 8, 2, 3), torch.tensor([1.0])
            rebuilt = model_patch_moss.head(model.model.diffusion_model,
                                           model.model.model_sampling.timestep(t).float(), feature, x)
            rebuilt = model.model.model_sampling.calculate_denoised(t, rebuilt.float(), x)
            torch.testing.assert_close(output, rebuilt, rtol=0, atol=0)
            model.model.fail = True
            with self.assertRaisesRegex(RuntimeError, "injected"):
                run_step(model, 0.9)
            self.assertEqual(len(model.model.diffusion_model.layers[-1]._forward_hooks), 0)
        run_sample(model, body=body)
        self.assertIsNone(model_patch._NATIVE_CAPTURE.get())

    def test_diffusion_modifier_before_or_after_spectrum_observes_final_features(self):
        for before_spectrum in (True, False):
            with self.subTest(before_spectrum=before_spectrum):
                model = Patcher()
                if not before_spectrum:
                    model = accelerated(model)
                model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL,
                                           "feature-modifier", feature_wrapper)
                if before_spectrum:
                    model = accelerated(model)
                block = model.model.diffusion_model.layers[-1]
                result = run_sample(model)
                self.assertEqual(model.model.calls, 5)
                self.assertEqual(len(result), 12)
                self.assertFalse(block._forward_hooks)

                def body():
                    output = run_step(model, 1.0)
                    feature = state_of(model).forecasters[0].h[-1]
                    x, t = torch.ones(1, 8, 2, 3), torch.tensor([1.0])
                    rebuilt = model_patch_moss.head(
                        model.model.diffusion_model,
                        model.model.model_sampling.timestep(t).float(), feature, x,
                    )
                    rebuilt = model.model.model_sampling.calculate_denoised(t, rebuilt.float(), x)
                    torch.testing.assert_close(output, rebuilt, rtol=0, atol=0)
                    with patch.object(block, "forward", side_effect=RuntimeError("injected block failure")):
                        with self.assertRaisesRegex(RuntimeError, "injected block failure"):
                            run_step(model, 0.9)
                    self.assertFalse(block._forward_hooks)
                    self.assertIsNone(model_patch._NATIVE_CAPTURE.get())
                    self.assertFalse(state_of(model).forecasters)

                run_sample(model, body=body)
                self.assertIsNone(model_patch._NATIVE_CAPTURE.get())

    def test_existing_explicit_cache_veto_is_preserved_on_downstream_clone(self):
        model = Patcher()
        model.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "feature-modifier", feature_wrapper)
        original = accelerated(model)
        clone = original.clone()
        clone.model_options["spectrum_cache_vetoes"] = [lambda **kwargs: False]
        original.model.current_patcher = clone
        run_sample(clone)
        self.assertEqual(original.model.calls, 12)
        original.model.current_patcher = original
        before = original.model.calls
        run_sample(original)
        self.assertEqual(original.model.calls - before, 5)
        self.assertFalse(original.model.diffusion_model.layers[-1]._forward_hooks)

    def test_real_core_compile_and_feature_hook_forecast_without_opt_in(self):
        from comfy_api.torch_helpers import set_torch_compile_wrapper

        model = Patcher()
        original_blocks = tuple(model.model.diffusion_model.layers)
        set_torch_compile_wrapper(model, backend="eager", dynamic=True,
                                  keys=[f"diffusion_model.layers.{i}" for i in range(2)])
        model.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "feature-modifier", feature_wrapper)
        model = accelerated(model)
        seen = []
        original_capture = model_patch_moss.capture
        def capture(dit):
            seen.append(type(dit.layers[-1]).__name__)
            return original_capture(dit)
        with patch.object(model_patch_moss, "capture", side_effect=capture):
            result = run_sample(model)
        self.assertEqual(model.model.calls, 5)
        self.assertTrue(seen and all(name == "OptimizedModule" for name in seen))
        self.assertTrue(all(torch.isfinite(x).all() for x in result))
        self.assertTrue(all(actual is original for actual, original in
                            zip(model.model.diffusion_model.layers, original_blocks)))
        self.assertTrue(all(not block._forward_hooks for block in original_blocks))
        self.assertIsNone(model_patch._NATIVE_CAPTURE.get())


if __name__ == "__main__":
    unittest.main()
