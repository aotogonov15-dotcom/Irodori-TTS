from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import gradio_app
import gradio_app_voicedesign
from irodori_tts.inference_runtime import (
    CharacterRuntimeSession,
    InferenceRuntime,
    RuntimeKey,
    RuntimeLifecycle,
    clear_cached_runtime,
)


class _FakeRuntime:
    def __init__(self, *, caption: bool) -> None:
        self.model_cfg = SimpleNamespace(
            use_caption_condition=caption,
            use_speaker_condition_resolved=True,
        )
        self._lifecycle_state = RuntimeLifecycle.BASE_READY
        self._runtime_owner_token = object()
        self.synthesize_runtime_ids: list[int] = []
        self.unload_count = 0

    @property
    def lifecycle_state(self) -> RuntimeLifecycle:
        return self._lifecycle_state

    def synthesize(self, request, *, log_fn):
        del request, log_fn
        self.synthesize_runtime_ids.append(id(self))
        self._lifecycle_state = RuntimeLifecycle.CHARACTER_LOCKED
        return SimpleNamespace(
            audio=torch.zeros(4),
            audios=[torch.zeros(4)],
            sample_rate=4,
            stage_timings=[],
            total_to_decode=0.0,
            used_seed=7,
            messages=[],
        )

    def unload(self) -> None:
        self.unload_count += 1
        self._lifecycle_state = RuntimeLifecycle.CLOSED


def _saved_path(path: str | Path, *_args, **_kwargs) -> Path:
    return Path(path)


class GradioRuntimeOwnershipTest(unittest.TestCase):
    def setUp(self) -> None:
        gradio_app._RUNTIME_SESSION.close()
        gradio_app_voicedesign._RUNTIME_SESSION.close()
        clear_cached_runtime()

    def tearDown(self) -> None:
        gradio_app._RUNTIME_SESSION.close()
        gradio_app_voicedesign._RUNTIME_SESSION.close()
        clear_cached_runtime()

    def test_reference_app_preload_and_two_generations_reuse_claimed_runtime(self) -> None:
        key = RuntimeKey(checkpoint="reference", model_device="cpu")
        first = _FakeRuntime(caption=False)
        replacement = _FakeRuntime(caption=False)
        session = CharacterRuntimeSession()
        generation_args = {
            "checkpoint": "ignored",
            "model_device": "cpu",
            "model_precision": "fp32",
            "codec_device": "cpu",
            "codec_precision": "fp32",
            "text": "hello",
            "uploaded_audio": None,
            "uploaded_speaker_embedding": None,
            "speaker_embedding_path_raw": "",
            "num_steps": 1,
            "num_candidates": 1,
            "seed_raw": "7",
            "seconds_raw": "0.5",
            "duration_scale": 1.0,
            "t_schedule_mode": "linear",
            "sway_coeff": -1.0,
            "cfg_guidance_mode": "independent",
            "cfg_scale_text": 3.0,
            "cfg_scale_speaker": 5.0,
            "cfg_scale_raw": "",
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "context_kv_cache": True,
            "truncation_factor_raw": "",
            "rescale_k_raw": "",
            "rescale_sigma_raw": "",
            "speaker_kv_scale_raw": "",
            "speaker_kv_min_t_raw": "",
            "speaker_kv_max_layers_raw": "",
            "lora_adapter_raw": "",
        }

        with (
            patch.object(gradio_app, "_RUNTIME_SESSION", session),
            patch.object(gradio_app, "_build_runtime_key", return_value=key),
            patch.object(gradio_app, "save_wav", side_effect=_saved_path),
            patch.object(
                InferenceRuntime,
                "from_key",
                side_effect=[first, replacement],
            ) as factory,
        ):
            gradio_app._load_model("ignored", "cpu", "fp32", "cpu", "fp32")
            gradio_app._run_generation(**generation_args)
            gradio_app._run_generation(**generation_args)

            self.assertEqual(first.synthesize_runtime_ids, [id(first), id(first)])
            factory.assert_called_once_with(key)

            owner_token = first._runtime_owner_token
            session_spec = session._spec
            status = gradio_app._load_model(
                "ignored", "cpu", "fp32", "cpu", "fp32"
            )
            self.assertIn("reused existing runtime", status)
            self.assertIs(session._runtime, first)
            self.assertIs(session._spec, session_spec)
            self.assertIs(first._runtime_owner_token, owner_token)
            self.assertEqual(first.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)
            self.assertEqual(first.unload_count, 0)
            factory.assert_called_once_with(key)

            gradio_app._run_generation(**generation_args)
            self.assertEqual(
                first.synthesize_runtime_ids,
                [id(first), id(first), id(first)],
            )
            factory.assert_called_once_with(key)

            gradio_app._run_generation(
                **{**generation_args, "lora_adapter_raw": "new-character"}
            )

        self.assertEqual(first.unload_count, 1)
        self.assertEqual(replacement.synthesize_runtime_ids, [id(replacement)])
        self.assertEqual(factory.call_count, 2)
        session.close()
        self.assertEqual(replacement.unload_count, 1)

    def test_voicedesign_preload_and_two_generations_reuse_claimed_runtime(self) -> None:
        key = RuntimeKey(checkpoint="voicedesign", model_device="cpu")
        runtime = _FakeRuntime(caption=True)
        session = CharacterRuntimeSession()
        generation_args = {
            "checkpoint": "ignored",
            "model_device": "cpu",
            "model_precision": "fp32",
            "codec_device": "cpu",
            "codec_precision": "fp32",
            "text": "hello",
            "caption": "bright",
            "ref_wav": None,
            "num_steps": 1,
            "num_candidates": 1,
            "seed_raw": "7",
            "seconds_raw": "0.5",
            "duration_scale": 1.0,
            "t_schedule_mode": "linear",
            "sway_coeff": -1.0,
            "cfg_guidance_mode": "independent",
            "cfg_scale_text": 3.0,
            "cfg_scale_caption": 4.0,
            "cfg_scale_speaker": 5.0,
            "cfg_scale_raw": "",
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "context_kv_cache": True,
            "speaker_kv_scale_raw": "",
            "max_text_len_raw": "",
            "max_caption_len_raw": "",
            "truncation_factor_raw": "",
            "rescale_k_raw": "",
            "rescale_sigma_raw": "",
            "lora_adapter_raw": "",
        }

        with (
            patch.object(gradio_app_voicedesign, "_RUNTIME_SESSION", session),
            patch.object(gradio_app_voicedesign, "_build_runtime_key", return_value=key),
            patch.object(gradio_app_voicedesign, "save_wav", side_effect=_saved_path),
            patch.object(InferenceRuntime, "from_key", return_value=runtime) as factory,
        ):
            gradio_app_voicedesign._describe_runtime(
                "ignored", "cpu", "fp32", "cpu", "fp32"
            )
            gradio_app_voicedesign._run_generation(**generation_args)
            gradio_app_voicedesign._run_generation(**generation_args)

            owner_token = runtime._runtime_owner_token
            session_spec = session._spec
            status = gradio_app_voicedesign._describe_runtime(
                "ignored", "cpu", "fp32", "cpu", "fp32"
            )
            self.assertIn("reused existing runtime", status)
            self.assertIs(session._runtime, runtime)
            self.assertIs(session._spec, session_spec)
            self.assertIs(runtime._runtime_owner_token, owner_token)
            self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)
            self.assertEqual(runtime.unload_count, 0)
            factory.assert_called_once_with(key)

            gradio_app_voicedesign._run_generation(**generation_args)

        self.assertEqual(
            runtime.synthesize_runtime_ids,
            [id(runtime), id(runtime), id(runtime)],
        )
        factory.assert_called_once_with(key)
        session.close()
        self.assertEqual(runtime.unload_count, 1)

    def test_reference_app_different_settings_preload_replaces_owned_runtime(self) -> None:
        first_key = RuntimeKey(checkpoint="reference-a", model_device="cpu")
        next_key = RuntimeKey(checkpoint="reference-b", model_device="cpu")
        first = _FakeRuntime(caption=False)
        replacement = _FakeRuntime(caption=False)
        session = CharacterRuntimeSession()
        generation_args = {
            "checkpoint": "ignored",
            "model_device": "cpu",
            "model_precision": "fp32",
            "codec_device": "cpu",
            "codec_precision": "fp32",
            "text": "hello",
            "uploaded_audio": None,
            "uploaded_speaker_embedding": None,
            "speaker_embedding_path_raw": "",
            "num_steps": 1,
            "num_candidates": 1,
            "seed_raw": "7",
            "seconds_raw": "0.5",
            "duration_scale": 1.0,
            "t_schedule_mode": "linear",
            "sway_coeff": -1.0,
            "cfg_guidance_mode": "independent",
            "cfg_scale_text": 3.0,
            "cfg_scale_speaker": 5.0,
            "cfg_scale_raw": "",
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "context_kv_cache": True,
            "truncation_factor_raw": "",
            "rescale_k_raw": "",
            "rescale_sigma_raw": "",
            "speaker_kv_scale_raw": "",
            "speaker_kv_min_t_raw": "",
            "speaker_kv_max_layers_raw": "",
            "lora_adapter_raw": "",
        }

        with (
            patch.object(gradio_app, "_RUNTIME_SESSION", session),
            patch.object(
                gradio_app,
                "_build_runtime_key",
                side_effect=[first_key, first_key, next_key, next_key],
            ),
            patch.object(gradio_app, "save_wav", side_effect=_saved_path),
            patch.object(
                InferenceRuntime,
                "from_key",
                side_effect=[first, replacement],
            ) as factory,
        ):
            gradio_app._load_model("ignored", "cpu", "fp32", "cpu", "fp32")
            gradio_app._run_generation(**generation_args)
            self.assertEqual(first.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)

            gradio_app._load_model("ignored", "cpu", "fp32", "cpu", "fp32")
            self.assertEqual(first.unload_count, 1)
            self.assertEqual(first.lifecycle_state, RuntimeLifecycle.CLOSED)
            self.assertIsNone(session._runtime)

            gradio_app._run_generation(**generation_args)

        self.assertEqual(factory.call_args_list[0].args, (first_key,))
        self.assertEqual(factory.call_args_list[1].args, (next_key,))
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(replacement.synthesize_runtime_ids, [id(replacement)])
        session.close()
        self.assertEqual(replacement.unload_count, 1)

    def test_voicedesign_different_settings_preload_replaces_owned_runtime(self) -> None:
        first_key = RuntimeKey(checkpoint="voicedesign-a", model_device="cpu")
        next_key = RuntimeKey(checkpoint="voicedesign-b", model_device="cpu")
        first = _FakeRuntime(caption=True)
        replacement = _FakeRuntime(caption=True)
        session = CharacterRuntimeSession()
        generation_args = {
            "checkpoint": "ignored",
            "model_device": "cpu",
            "model_precision": "fp32",
            "codec_device": "cpu",
            "codec_precision": "fp32",
            "text": "hello",
            "caption": "bright",
            "ref_wav": None,
            "num_steps": 1,
            "num_candidates": 1,
            "seed_raw": "7",
            "seconds_raw": "0.5",
            "duration_scale": 1.0,
            "t_schedule_mode": "linear",
            "sway_coeff": -1.0,
            "cfg_guidance_mode": "independent",
            "cfg_scale_text": 3.0,
            "cfg_scale_caption": 4.0,
            "cfg_scale_speaker": 5.0,
            "cfg_scale_raw": "",
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "context_kv_cache": True,
            "speaker_kv_scale_raw": "",
            "max_text_len_raw": "",
            "max_caption_len_raw": "",
            "truncation_factor_raw": "",
            "rescale_k_raw": "",
            "rescale_sigma_raw": "",
            "lora_adapter_raw": "",
        }

        with (
            patch.object(gradio_app_voicedesign, "_RUNTIME_SESSION", session),
            patch.object(
                gradio_app_voicedesign,
                "_build_runtime_key",
                side_effect=[first_key, first_key, next_key, next_key],
            ),
            patch.object(
                gradio_app_voicedesign,
                "save_wav",
                side_effect=_saved_path,
            ),
            patch.object(
                InferenceRuntime,
                "from_key",
                side_effect=[first, replacement],
            ) as factory,
        ):
            gradio_app_voicedesign._describe_runtime(
                "ignored", "cpu", "fp32", "cpu", "fp32"
            )
            gradio_app_voicedesign._run_generation(**generation_args)
            self.assertEqual(first.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)

            gradio_app_voicedesign._describe_runtime(
                "ignored", "cpu", "fp32", "cpu", "fp32"
            )
            self.assertEqual(first.unload_count, 1)
            self.assertEqual(first.lifecycle_state, RuntimeLifecycle.CLOSED)
            self.assertIsNone(session._runtime)

            gradio_app_voicedesign._run_generation(**generation_args)

        self.assertEqual(factory.call_args_list[0].args, (first_key,))
        self.assertEqual(factory.call_args_list[1].args, (next_key,))
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(replacement.synthesize_runtime_ids, [id(replacement)])
        session.close()
        self.assertEqual(replacement.unload_count, 1)

    def test_unload_callback_closes_owned_runtime(self) -> None:
        key = RuntimeKey(checkpoint="shutdown", model_device="cpu")
        runtime = _FakeRuntime(caption=False)
        session = CharacterRuntimeSession()
        with (
            patch.object(gradio_app, "_RUNTIME_SESSION", session),
            patch.object(InferenceRuntime, "from_key", return_value=runtime),
        ):
            session.acquire(key, lora_adapter=None)
            gradio_app._clear_runtime_cache()

        self.assertEqual(runtime.unload_count, 1)
        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.CLOSED)


if __name__ == "__main__":
    unittest.main()
