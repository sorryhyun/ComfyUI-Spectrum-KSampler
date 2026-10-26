"""Per-call block replacements must not share native feature forecasts."""
import unittest
import weakref

from test_native_moss_spectrum import (
    accelerated, model_patch, run_sample, run_step, spectrum, state_of, torch,
)


def skip(args, extra_args):
    return args


def options(branch=0, *, layer=None, replacement=skip):
    value = {"uuids": [f"conditioning-{branch}"]}
    if layer is not None:
        value["patches_replace"] = {"dit": {("single_block", layer): replacement}}
    return value


class ModelPatchContextsTest(unittest.TestCase):
    def test_same_cond_and_sigma_keep_regular_and_slg_histories_separate(self):
        model = accelerated(compat_policy="strict")

        def body():
            for step, sigma in enumerate((1.0, 0.9, 0.8)):
                regular = run_step(model, sigma, options=options())
                regular_history = state_of(model).forecasters[(0, "conditioning-0")]
                last_regular = regular_history.h[-1].clone()
                slg = run_step(model, sigma, options=options(layer=0))
                self.assertFalse(torch.equal(regular, slg))
                self.assertEqual(len(state_of(model).forecasters), 2)
                self.assertEqual(regular_history.t, list(map(float, range(step + 1))))
                torch.testing.assert_close(regular_history.h[-1], last_regular, rtol=0, atol=0)
            self.assertEqual(model.model.calls, 6)
            regular = run_step(model, 0.7, options=options())
            slg = run_step(model, 0.7, options=options(layer=0))
            self.assertEqual(model.model.calls, 6)
            self.assertFalse(torch.equal(regular, slg))

        run_sample(model, body=body)

    def test_late_slg_context_collects_its_own_observations(self):
        model = accelerated()

        def body():
            for sigma in (1.0, 0.9, 0.8):
                run_step(model, sigma, options=options())
            for sigma in (0.7, 0.6, 0.5):
                before = model.model.calls
                run_step(model, sigma, options=options())
                self.assertEqual(model.model.calls, before)
                run_step(model, sigma, options=options(layer=0))
                self.assertEqual(model.model.calls, before + 1)
                self.assertEqual(state_of(model).mode, "actual")
            before = model.model.calls
            run_step(model, 0.4, options=options())
            run_step(model, 0.4, options=options(layer=0))
            self.assertEqual(model.model.calls, before)

        run_sample(model, body=body)

    def test_uncond_slg_window_keeps_patched_and_unpatched_histories(self):
        model = accelerated()

        def body():
            for sigma in (1.0, 0.9, 0.8):
                for branch in (0, 1):
                    run_step(model, sigma, branch=branch, options=options(branch))
            uncond = state_of(model).forecasters[(1, "conditioning-1")]
            original_history = [value.clone() for value in uncond.h]
            for sigma in (0.7, 0.6, 0.5):
                before = model.model.calls
                run_step(model, sigma, options=options())
                run_step(model, sigma, branch=1, options=options(1, layer=0))
                self.assertEqual(model.model.calls, before + 1)
            before = model.model.calls
            run_step(model, 0.4, options=options())
            run_step(model, 0.4, branch=1, options=options(1, layer=0))
            # Leaving the window returns to the original uncond history.
            run_step(model, 0.3, options=options())
            run_step(model, 0.3, branch=1, options=options(1))
            self.assertEqual(model.model.calls, before)
            self.assertEqual(uncond.t, [0.0, 1.0, 2.0])
            for actual, expected in zip(uncond.h, original_history):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(len(state_of(model).forecasters), 3)

        run_sample(model, body=body)

    def test_unsupported_post_cfg_capture_records_actual_and_rewarms(self):
        model = accelerated()

        def body():
            for sigma in (1.0, 0.9, 0.8):
                run_step(model, sigma, options=options())
            run_step(model, 0.7, options=options())
            self.assertEqual(model.model.calls, 3)
            # Skipping the final MOSS block bypasses its capture hook entirely.
            run_step(model, 0.7, options=options(layer=1))
            self.assertEqual(model.model.calls, 4)
            self.assertEqual(state_of(model).mode, "actual")
            self.assertFalse(state_of(model).forecasters)
            for sigma in (0.6, 0.5, 0.4):
                run_step(model, sigma, options=options())
            self.assertEqual(model.model.calls, 7)
            run_step(model, 0.3, options=options())
            self.assertEqual(model.model.calls, 7)

        run_sample(model, body=body)
        self.assertIsNone(model_patch._NATIVE_CAPTURE.get())
        self.assertFalse(model.model.diffusion_model.layers[-1]._forward_hooks)

    def test_callable_identity_is_snapshot_scoped_and_preserves_uuid_policy(self):
        class Replacement:
            __hash__ = None

            def __call__(self, args, extra_args):
                return args

        model = accelerated(compat_policy="strict")
        references, states = [], []

        def body():
            state = state_of(model)
            states.append(state)
            replacement = Replacement()
            references.append(weakref.ref(replacement))
            c = {"transformer_options": options(layer=0, replacement=replacement)}
            first = model_patch._model_patch_batch_keys(state, c, [0])
            self.assertTrue(spectrum._uses_uuid_branch_keys(first))
            self.assertEqual(first, model_patch._model_patch_batch_keys(state, c, [0]))
            c["transformer_options"]["patches_replace"]["dit"][("single_block", 0)] = Replacement()
            second = model_patch._model_patch_batch_keys(state, c, [0])
            self.assertNotEqual(first, second)
            del replacement
            self.assertIsNotNone(references[0]())
            del c["transformer_options"]["uuids"]
            self.assertFalse(spectrum._uses_uuid_branch_keys(
                model_patch._model_patch_batch_keys(state, c, [0])))
            self.assertEqual(model_patch._model_patch_batch_keys(state, {}, [0, 1]), [0, 1])

        run_sample(model, body=body)
        self.assertIsNone(references[0]())
        self.assertFalse(states[0].native_patch_callables)


if __name__ == "__main__":
    unittest.main()
