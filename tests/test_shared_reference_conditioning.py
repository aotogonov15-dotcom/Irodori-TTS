from __future__ import annotations

import math
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    PreparedReferenceConditioning,
    SamplingRequest,
)
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.rf import sample_euler_rf_cfg


class _FakeTokenizer:
    def batch_encode(self, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        del max_length
        batch_size = len(texts)
        return (
            torch.ones((batch_size, 3), dtype=torch.long),
            torch.ones((batch_size, 3), dtype=torch.bool),
        )


class _FakeCodec:
    def __init__(self) -> None:
        self.sample_rate = 4
        self.model = SimpleNamespace(hop_length=2)
        self.encode_count = 0

    def encode_waveform(self, waveform: torch.Tensor, **kwargs) -> torch.Tensor:
        del waveform, kwargs
        self.encode_count += 1
        return torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros((latent.shape[0], 1, latent.shape[1] * 2), dtype=torch.float32)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(
            patched_latent_dim=2,
            use_speaker_condition_resolved=True,
            use_caption_condition=False,
        )
        self.speaker_encode_count = 0
        self.condition_speaker_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.forward_speaker_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    def encode_speaker_condition(
        self,
        ref_latent: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
        *,
        batch_size: int,
        speaker_state_override: torch.Tensor | None = None,
        speaker_mask_override: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ref_mask, speaker_mask_override
        if speaker_state_override is not None:
            raise AssertionError("prepared inputs must not be re-encoded")
        self.speaker_encode_count += 1
        assert ref_latent is not None
        state = torch.arange(12, dtype=dtype, device=device).reshape(1, 4, 3)
        mask = torch.ones((1, 4), dtype=torch.bool, device=device)
        self.assert_batch_size = batch_size
        return state, mask

    def encode_conditions(self, **kwargs):
        state = kwargs["speaker_state_override"]
        mask = kwargs["speaker_mask_override"]
        if state is None or mask is None:
            raise AssertionError("duration and sampling must receive prepared conditioning")
        self.condition_speaker_inputs.append((state, mask))
        text_ids = kwargs["text_input_ids"]
        text_mask = kwargs["text_mask"]
        text_state = torch.zeros((*text_ids.shape, 3), dtype=self.weight.dtype)
        return text_state, text_mask, state, mask, None, None

    def predict_duration_log_frames(self, **kwargs) -> torch.Tensor:
        batch_size = kwargs["text_state"].shape[0]
        return torch.full((batch_size,), math.log1p(4.0), dtype=torch.float32)

    def forward_with_encoded_conditions(self, **kwargs) -> torch.Tensor:
        self.forward_speaker_inputs.append((kwargs["speaker_state"], kwargs["speaker_mask"]))
        return torch.zeros_like(kwargs["x_t"])


def _fake_sampler(**kwargs) -> torch.Tensor:
    model = kwargs["model"]
    model.encode_conditions(
        text_input_ids=kwargs["text_input_ids"],
        text_mask=kwargs["text_mask"],
        ref_latent=kwargs["ref_latent"],
        ref_mask=kwargs["ref_mask"],
        caption_input_ids=kwargs["caption_input_ids"],
        caption_mask=kwargs["caption_mask"],
        speaker_state_override=kwargs["speaker_state_override"],
        speaker_mask_override=kwargs["speaker_mask_override"],
    )
    return torch.zeros(
        (
            kwargs["text_input_ids"].shape[0],
            kwargs["sequence_length"],
            model.cfg.patched_latent_dim,
        ),
        dtype=model.weight.dtype,
    )


def _make_runtime() -> InferenceRuntime:
    runtime = object.__new__(InferenceRuntime)
    runtime.key = SimpleNamespace(
        model_device="cpu",
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
    )
    runtime.model_device = torch.device("cpu")
    runtime.codec_device = torch.device("cpu")
    runtime.model_cfg = SimpleNamespace(
        use_speaker_condition_resolved=True,
        use_caption_condition=False,
        use_duration_predictor=True,
        speaker_patch_size=1,
        speaker_dim=3,
        latent_dim=2,
        latent_patch_size=1,
    )
    runtime.train_cfg = None
    runtime.model = _FakeModel()
    runtime.tokenizer = _FakeTokenizer()
    runtime.caption_tokenizer = None
    runtime.codec = _FakeCodec()
    runtime.default_text_max_len = 16
    runtime.default_caption_max_len = 16
    runtime.watermarker = SimpleNamespace(ready=False)
    runtime._infer_lock = threading.Lock()
    runtime._model_dtype = torch.float32
    runtime._lora_adapter_names = {}
    return runtime


class PreparedReferenceConditioningTest(unittest.TestCase):
    def test_expands_batch_as_views_without_mutating_base_tensors(self) -> None:
        state = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        mask = torch.tensor([[True, True, False, True]])
        prepared = PreparedReferenceConditioning(state, mask)

        expanded_state, expanded_mask = prepared.expanded(3)

        self.assertEqual(expanded_state.shape, (3, 4, 3))
        self.assertEqual(expanded_mask.shape, (3, 4))
        self.assertEqual(
            expanded_state.untyped_storage().data_ptr(), state.untyped_storage().data_ptr()
        )
        self.assertEqual(
            expanded_mask.untyped_storage().data_ptr(), mask.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(prepared.speaker_state, state)
        torch.testing.assert_close(prepared.speaker_mask, mask)

    def test_model_conditioning_is_numerically_equivalent_and_encoded_once(self) -> None:
        torch.manual_seed(7)
        cfg = ModelConfig(
            latent_dim=2,
            latent_patch_size=1,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=2.0,
            text_mlp_ratio=2.0,
            speaker_mlp_ratio=2.0,
            dropout=0.0,
            text_vocab_size=16,
            text_dim=8,
            text_layers=1,
            text_heads=2,
            use_speaker_condition=True,
            speaker_dim=8,
            speaker_layers=1,
            speaker_heads=2,
            timestep_embed_dim=8,
            adaln_rank=4,
        )
        model = TextToLatentRFDiT(cfg).eval()
        text_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        text_mask = torch.ones_like(text_ids, dtype=torch.bool)
        ref_latent = torch.randn((1, 6, 2))
        ref_mask = torch.tensor([[True, True, True, True, True, False]])

        with torch.inference_mode():
            legacy_state, legacy_mask = model.encode_conditions(
                text_ids, text_mask, ref_latent, ref_mask
            )[2:4]
            with patch.object(
                model.speaker_encoder,
                "forward",
                wraps=model.speaker_encoder.forward,
            ) as speaker_forward:
                prepared_state, prepared_mask = model.encode_speaker_condition(
                    ref_latent,
                    ref_mask,
                    batch_size=1,
                )
                duration_state, duration_mask = model.encode_conditions(
                    text_ids,
                    text_mask,
                    None,
                    None,
                    speaker_state_override=prepared_state,
                    speaker_mask_override=prepared_mask,
                )[2:4]
                sampling_state, sampling_mask = model.encode_conditions(
                    text_ids,
                    text_mask,
                    None,
                    None,
                    speaker_state_override=prepared_state,
                    speaker_mask_override=prepared_mask,
                )[2:4]

        self.assertEqual(speaker_forward.call_count, 1)
        torch.testing.assert_close(prepared_state, legacy_state, rtol=0, atol=0)
        torch.testing.assert_close(duration_state, legacy_state, rtol=0, atol=0)
        torch.testing.assert_close(sampling_state, legacy_state, rtol=0, atol=0)
        torch.testing.assert_close(prepared_mask, legacy_mask)
        torch.testing.assert_close(duration_mask, legacy_mask)
        torch.testing.assert_close(sampling_mask, legacy_mask)

    def test_cfg_expansion_does_not_mutate_prepared_batch_views(self) -> None:
        model = _FakeModel()
        state = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        mask = torch.tensor([[True, True, False, True]])
        prepared = PreparedReferenceConditioning(state, mask)
        expanded_state, expanded_mask = prepared.expanded(2)
        original_state = state.clone()
        original_mask = mask.clone()

        output = sample_euler_rf_cfg(
            model=model,
            text_input_ids=torch.ones((2, 3), dtype=torch.long),
            text_mask=torch.ones((2, 3), dtype=torch.bool),
            ref_latent=None,
            ref_mask=None,
            sequence_length=2,
            speaker_state_override=expanded_state,
            speaker_mask_override=expanded_mask,
            num_steps=1,
            cfg_scale_text=3.0,
            cfg_scale_speaker=5.0,
            seed=9,
            use_context_kv_cache=False,
        )

        self.assertEqual(output.shape, (2, 2, 2))
        self.assertEqual(model.forward_speaker_inputs[0][0].shape, (6, 4, 3))
        self.assertEqual(model.forward_speaker_inputs[0][1].shape, (6, 4))
        torch.testing.assert_close(prepared.speaker_state, original_state)
        torch.testing.assert_close(prepared.speaker_mask, original_mask)


class InferenceRuntimeSharedConditioningTest(unittest.TestCase):
    def test_wav_path_prepares_once_and_reuses_same_state_for_duration_and_sampling(self) -> None:
        runtime = _make_runtime()
        request = SamplingRequest(
            text="shared conditioning",
            ref_wav="reference.wav",
            ref_normalize_db=None,
            num_candidates=2,
            num_steps=1,
            seed=1234,
            trim_tail=False,
        )

        with (
            patch(
                "irodori_tts.inference_runtime._load_audio",
                return_value=(torch.zeros((1, 8)), 4),
            ) as load_audio,
            patch("irodori_tts.inference_runtime.sample_euler_rf_cfg", _fake_sampler),
        ):
            result = runtime.synthesize(request)

        self.assertEqual(load_audio.call_count, 1)
        self.assertEqual(runtime.codec.encode_count, 1)
        self.assertEqual(runtime.model.speaker_encode_count, 1)
        self.assertEqual(len(runtime.model.condition_speaker_inputs), 2)
        duration_state, duration_mask = runtime.model.condition_speaker_inputs[0]
        sampling_state, sampling_mask = runtime.model.condition_speaker_inputs[1]
        self.assertIs(duration_state, sampling_state)
        self.assertIs(duration_mask, sampling_mask)
        self.assertEqual(duration_state.shape, (2, 4, 3))
        self.assertEqual(result.audio.shape, (1, 8))
        self.assertEqual(result.used_seed, 1234)

    def test_prepared_override_skips_wav_decode_and_reference_encode_without_mutation(self) -> None:
        runtime = _make_runtime()
        with patch(
            "irodori_tts.inference_runtime._load_audio",
            return_value=(torch.zeros((1, 8)), 4),
        ):
            prepared = runtime.prepare_reference_conditioning(
                ref_wav="reference.wav",
                ref_normalize_db=None,
            )
        original_state = prepared.speaker_state.clone()
        original_mask = prepared.speaker_mask.clone()

        request = SamplingRequest(
            text="prepared override",
            num_candidates=2,
            num_steps=1,
            seed=4321,
            trim_tail=False,
        )
        with (
            patch(
                "irodori_tts.inference_runtime._load_audio",
                side_effect=AssertionError("prepared path must not decode WAV"),
            ),
            patch("irodori_tts.inference_runtime.sample_euler_rf_cfg", _fake_sampler),
        ):
            result = runtime.synthesize(request, prepared_reference=prepared)

        self.assertEqual(runtime.codec.encode_count, 1)
        self.assertEqual(runtime.model.speaker_encode_count, 1)
        torch.testing.assert_close(prepared.speaker_state, original_state)
        torch.testing.assert_close(prepared.speaker_mask, original_mask)
        self.assertEqual(result.used_seed, 4321)


if __name__ == "__main__":
    unittest.main()
