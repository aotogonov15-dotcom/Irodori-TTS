from __future__ import annotations

import json
import math
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import (
    CharacterRuntimeSession,
    InferenceRuntime,
    PreparedReferenceConditioning,
    RuntimeKey,
    RuntimeLifecycle,
    SamplingRequest,
    claim_cached_runtime,
    clear_cached_runtime,
    preload_cached_runtime,
)
from irodori_tts.model import TextToLatentRFDiT, patch_sequence_with_mask
from irodori_tts.rf import sample_euler_rf_cfg
from irodori_tts.speaker_inversion import save_speaker_inversion_safetensors


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
        self.duration_has_speaker: list[torch.Tensor] = []

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
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        del speaker_mask_override
        if speaker_state_override is not None:
            raise AssertionError("prepared inputs must not be re-encoded")
        if not self.cfg.use_speaker_condition_resolved:
            return None, None
        self.speaker_encode_count += 1
        assert ref_latent is not None and ref_mask is not None
        state = torch.arange(12, dtype=dtype, device=device).reshape(1, 4, 3)
        mask = torch.full(
            (1, 4),
            bool(ref_mask.any().item()),
            dtype=torch.bool,
            device=device,
        )
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
        self.duration_has_speaker.append(kwargs["has_speaker"].detach().clone())
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


def _zero_sampler(**kwargs) -> torch.Tensor:
    return torch.zeros(
        (
            kwargs["text_input_ids"].shape[0],
            kwargs["sequence_length"],
            kwargs["model"].cfg.patched_latent_dim,
        ),
        dtype=next(kwargs["model"].parameters()).dtype,
    )


def _small_model_config(*, use_duration_predictor: bool = False) -> ModelConfig:
    return ModelConfig(
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
        use_duration_predictor=use_duration_predictor,
        duration_hidden_dim=8,
        duration_layers=1,
        duration_dropout=0.0,
        duration_attention_heads=2,
    )


def _legacy_speaker_condition_oracle(
    model: TextToLatentRFDiT,
    ref_latent: torch.Tensor,
    ref_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Test-only copy of the pre-B1 reference path from Base 47b62bd."""
    patched, patched_mask = patch_sequence_with_mask(
        seq=ref_latent,
        mask=ref_mask,
        patch_size=model.cfg.speaker_patch_size,
    )
    state = model.speaker_encoder(patched, patched_mask)
    state = model.speaker_norm(state)
    mask_f = patched_mask.unsqueeze(-1).to(dtype=state.dtype)
    mean = (state * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    has_any = patched_mask.any(dim=1, keepdim=True)
    return torch.cat([mean, state], dim=1), torch.cat([has_any, patched_mask], dim=1)


def _make_runtime() -> InferenceRuntime:
    runtime = object.__new__(InferenceRuntime)
    runtime.key = SimpleNamespace(
        model_device="cpu",
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
        compile_model=False,
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
    runtime._lifecycle_state = RuntimeLifecycle.BASE_READY
    runtime._runtime_owner_token = object()
    runtime._character_lora_adapter = None
    runtime._character_adapter_name = None
    runtime._failure_reason = None
    return runtime


def _make_real_runtime(
    *,
    cfg: ModelConfig,
    state_dict: dict[str, torch.Tensor],
) -> InferenceRuntime:
    runtime = object.__new__(InferenceRuntime)
    runtime.key = SimpleNamespace(
        model_device="cpu",
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
        compile_model=False,
    )
    runtime.model_device = torch.device("cpu")
    runtime.codec_device = torch.device("cpu")
    runtime.model_cfg = cfg
    runtime.train_cfg = None
    runtime.model = TextToLatentRFDiT(cfg).eval()
    runtime.model.load_state_dict(state_dict)
    runtime.tokenizer = _FakeTokenizer()
    runtime.caption_tokenizer = None
    runtime.codec = _FakeCodec()
    runtime.default_text_max_len = 16
    runtime.default_caption_max_len = 16
    runtime.watermarker = SimpleNamespace(ready=False)
    runtime._infer_lock = threading.Lock()
    runtime._model_dtype = torch.float32
    runtime._lifecycle_state = RuntimeLifecycle.BASE_READY
    runtime._runtime_owner_token = object()
    runtime._character_lora_adapter = None
    runtime._character_adapter_name = None
    runtime._failure_reason = None
    return runtime


class PreparedReferenceConditioningTest(unittest.TestCase):
    def test_rejects_empty_speaker_token_sequence(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one speaker token"):
            PreparedReferenceConditioning(
                torch.empty((1, 0, 8)),
                torch.empty((1, 0), dtype=torch.bool),
                _lora_adapter=None,
            )

    def test_expands_batch_as_views_without_mutating_base_tensors(self) -> None:
        state = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        mask = torch.tensor([[True, True, False, True]])
        prepared = PreparedReferenceConditioning(state, mask, _lora_adapter=None)

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

    def test_disabled_speaker_conditioning_requires_an_inactive_mask(self) -> None:
        state = torch.zeros((1, 2, 8))
        with self.assertRaisesRegex(ValueError, "no active speaker tokens"):
            PreparedReferenceConditioning(
                state,
                torch.tensor([[True, False]]),
                _lora_adapter=None,
                speaker_conditioning_enabled=False,
            )

        prepared = PreparedReferenceConditioning(
            state,
            torch.zeros((1, 2), dtype=torch.bool),
            _lora_adapter=None,
            speaker_conditioning_enabled=False,
        )
        self.assertFalse(prepared.speaker_mask.any().item())

    def test_enabled_speaker_conditioning_requires_an_active_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "speaker_state and speaker_mask to be set"):
            PreparedReferenceConditioning(
                None,
                None,
                _lora_adapter=None,
                speaker_conditioning_enabled=True,
            )

        state = torch.zeros((1, 2, 8))
        with self.assertRaisesRegex(ValueError, "at least one active speaker token"):
            PreparedReferenceConditioning(
                state,
                torch.zeros((1, 2), dtype=torch.bool),
                _lora_adapter=None,
                speaker_conditioning_enabled=True,
            )

        prepared = PreparedReferenceConditioning(
            state,
            torch.tensor([[True, False]]),
            _lora_adapter=None,
            speaker_conditioning_enabled=True,
        )
        self.assertTrue(prepared.speaker_mask.any().item())

    def test_disabled_speaker_conditioning_allows_none_no_ref_state(self) -> None:
        prepared = PreparedReferenceConditioning(
            None,
            None,
            _lora_adapter=None,
            speaker_conditioning_enabled=False,
        )
        self.assertIsNone(prepared.speaker_state)
        self.assertIsNone(prepared.speaker_mask)

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

    def test_independent_base_oracle_covers_reference_no_ref_and_direct_inputs(self) -> None:
        torch.manual_seed(17)
        model = TextToLatentRFDiT(_small_model_config()).eval()
        reference_cases = (
            (
                torch.randn((1, 6, 2)),
                torch.tensor([[True, True, True, True, True, False]]),
            ),
            (
                torch.zeros((1, 1, 2)),
                torch.zeros((1, 1), dtype=torch.bool),
            ),
        )

        with torch.inference_mode():
            for ref_latent, ref_mask in reference_cases:
                expected_state, expected_mask = _legacy_speaker_condition_oracle(
                    model, ref_latent, ref_mask
                )
                actual_state, actual_mask = model.encode_speaker_condition(
                    ref_latent,
                    ref_mask,
                    batch_size=1,
                )
                torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
                torch.testing.assert_close(actual_mask, expected_mask)

            direct_state = torch.randn((1, 3, 8))
            direct_mask = torch.tensor([[True, False, True]])
            actual_state, actual_mask = model.encode_speaker_condition(
                None,
                None,
                batch_size=2,
                speaker_state_override=direct_state,
                speaker_mask_override=direct_mask,
            )
            torch.testing.assert_close(actual_state, direct_state.expand(2, -1, -1))
            torch.testing.assert_close(actual_mask, direct_mask.expand(2, -1))

    def test_independent_oracle_covers_speaker_inversion(self) -> None:
        torch.manual_seed(19)
        model = TextToLatentRFDiT(_small_model_config()).eval()
        initial = torch.randn((3, 8))
        model.enable_speaker_inversion(
            num_tokens=3,
            init_std=0.0,
            init_embedding=initial,
        )

        with torch.inference_mode():
            actual_state, actual_mask = model.encode_speaker_condition(
                None,
                None,
                batch_size=2,
            )

        torch.testing.assert_close(actual_state, initial.unsqueeze(0).expand(2, -1, -1))
        torch.testing.assert_close(actual_mask, torch.ones((2, 3), dtype=torch.bool))

    def test_real_duration_and_rf_kv_paths_preserve_prepared_tensors_and_equivalence(self) -> None:
        torch.manual_seed(23)
        model = TextToLatentRFDiT(
            _small_model_config(use_duration_predictor=True)
        ).eval()
        torch.nn.init.normal_(model.out_proj.weight, std=0.02)
        ref_latent = torch.randn((1, 6, 2))
        ref_mask = torch.tensor([[True, True, True, True, True, False]])
        text_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        text_mask = torch.ones_like(text_ids, dtype=torch.bool)

        with torch.inference_mode():
            prepared_state, prepared_mask = model.encode_speaker_condition(
                ref_latent,
                ref_mask,
                batch_size=1,
            )
            prepared = PreparedReferenceConditioning(
                prepared_state.detach(),
                prepared_mask.detach(),
                _lora_adapter=None,
            )
            original_state = prepared.speaker_state.clone()
            original_mask = prepared.speaker_mask.clone()

            legacy_conditions = model.encode_conditions(
                text_ids,
                text_mask,
                ref_latent,
                ref_mask,
            )
            prepared_conditions = model.encode_conditions(
                text_ids,
                text_mask,
                None,
                None,
                speaker_state_override=prepared.speaker_state,
                speaker_mask_override=prepared.speaker_mask,
            )
            duration_kwargs = {
                "duration_features": torch.zeros((1, model.cfg.duration_aux_dim)),
                "has_speaker": torch.ones((1,), dtype=torch.bool),
            }
            legacy_duration = model.predict_duration_log_frames(
                text_state=legacy_conditions[0],
                text_mask=legacy_conditions[1],
                speaker_state=legacy_conditions[2],
                speaker_mask=legacy_conditions[3],
                **duration_kwargs,
            )
            prepared_duration = model.predict_duration_log_frames(
                text_state=prepared_conditions[0],
                text_mask=prepared_conditions[1],
                speaker_state=prepared_conditions[2],
                speaker_mask=prepared_conditions[3],
                **duration_kwargs,
            )
            legacy_sample = sample_euler_rf_cfg(
                model=model,
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                sequence_length=2,
                num_steps=2,
                cfg_scale_text=3.0,
                cfg_scale_speaker=5.0,
                seed=29,
                use_context_kv_cache=True,
            )
            prepared_sample = sample_euler_rf_cfg(
                model=model,
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=None,
                ref_mask=None,
                sequence_length=2,
                speaker_state_override=prepared.speaker_state,
                speaker_mask_override=prepared.speaker_mask,
                num_steps=2,
                cfg_scale_text=3.0,
                cfg_scale_speaker=5.0,
                seed=29,
                use_context_kv_cache=True,
            )

        torch.testing.assert_close(prepared_duration, legacy_duration, rtol=0, atol=0)
        torch.testing.assert_close(prepared_sample, legacy_sample, rtol=0, atol=0)
        torch.testing.assert_close(prepared.speaker_state, original_state, rtol=0, atol=0)
        torch.testing.assert_close(prepared.speaker_mask, original_mask)

    def test_real_cfg_off_no_ref_duration_and_sampling_match_prepared_strictly(self) -> None:
        torch.manual_seed(67)
        model = TextToLatentRFDiT(
            _small_model_config(use_duration_predictor=True)
        ).eval()
        torch.nn.init.normal_(model.out_proj.weight, std=0.02)
        no_ref_latent = torch.zeros((1, 1, 2))
        no_ref_mask = torch.zeros((1, 1), dtype=torch.bool)
        text_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        text_mask = torch.ones_like(text_ids, dtype=torch.bool)

        with torch.inference_mode():
            prepared_state, prepared_mask = model.encode_speaker_condition(
                no_ref_latent,
                no_ref_mask,
                batch_size=1,
            )
            ordinary_conditions = model.encode_conditions(
                text_ids,
                text_mask,
                no_ref_latent,
                no_ref_mask,
            )
            prepared_conditions = model.encode_conditions(
                text_ids,
                text_mask,
                None,
                None,
                speaker_state_override=prepared_state,
                speaker_mask_override=prepared_mask,
            )
            duration_kwargs = {
                "duration_features": torch.zeros((1, model.cfg.duration_aux_dim)),
                "has_speaker": torch.zeros((1,), dtype=torch.bool),
            }
            ordinary_duration = model.predict_duration_log_frames(
                text_state=ordinary_conditions[0],
                text_mask=ordinary_conditions[1],
                speaker_state=ordinary_conditions[2],
                speaker_mask=ordinary_conditions[3],
                **duration_kwargs,
            )
            prepared_duration = model.predict_duration_log_frames(
                text_state=prepared_conditions[0],
                text_mask=prepared_conditions[1],
                speaker_state=prepared_conditions[2],
                speaker_mask=prepared_conditions[3],
                **duration_kwargs,
            )
            ordinary_sample = sample_euler_rf_cfg(
                model=model,
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=no_ref_latent,
                ref_mask=no_ref_mask,
                sequence_length=2,
                num_steps=2,
                cfg_scale=0.0,
                seed=71,
                use_context_kv_cache=True,
            )
            prepared_sample = sample_euler_rf_cfg(
                model=model,
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=None,
                ref_mask=None,
                sequence_length=2,
                speaker_state_override=prepared_state,
                speaker_mask_override=prepared_mask,
                num_steps=2,
                cfg_scale=0.0,
                seed=71,
                use_context_kv_cache=True,
            )

        self.assertFalse(prepared_mask.any().item())
        for prepared_tensor, ordinary_tensor in zip(
            prepared_conditions,
            ordinary_conditions,
            strict=True,
        ):
            if prepared_tensor is not None:
                torch.testing.assert_close(
                    prepared_tensor, ordinary_tensor, rtol=0, atol=0
                )
        torch.testing.assert_close(prepared_duration, ordinary_duration, rtol=0, atol=0)
        torch.testing.assert_close(prepared_sample, ordinary_sample, rtol=0, atol=0)

    def test_cfg_expansion_does_not_mutate_prepared_batch_views(self) -> None:
        model = _FakeModel()
        state = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        mask = torch.tensor([[True, True, False, True]])
        prepared = PreparedReferenceConditioning(state, mask, _lora_adapter=None)
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

    def test_prepared_no_ref_matches_ordinary_no_ref_across_cfg_duration_and_kv(self) -> None:
        cases = (
            ("cfg_off_manual_kv_off", "independent", 0.0, 0.5, False, None),
            ("cfg_joint_auto_kv_scaled", "joint", None, None, True, 2.0),
            ("cfg_independent_auto_kv_scaled", "independent", None, None, True, 2.0),
        )
        for name, cfg_mode, cfg_scale, seconds, context_kv_cache, speaker_kv_scale in cases:
            with self.subTest(name=name):
                ordinary_runtime = _make_runtime()
                prepared_runtime = _make_runtime()
                prepared = prepared_runtime.prepare_reference_conditioning(no_ref=True)
                self.assertFalse(prepared.speaker_conditioning_enabled)
                self.assertFalse(prepared.speaker_mask.any().item())

                common = {
                    "text": "prepared no reference",
                    "seconds": seconds,
                    "min_seconds": 0.5,
                    "max_seconds": 2.0,
                    "num_steps": 1,
                    "seed": 53,
                    "trim_tail": False,
                    "cfg_guidance_mode": cfg_mode,
                    "cfg_scale": cfg_scale,
                    "context_kv_cache": context_kv_cache,
                    "speaker_kv_scale": speaker_kv_scale,
                }
                ordinary_request = SamplingRequest(no_ref=True, **common)
                prepared_request = SamplingRequest(**common)
                sampler_calls: list[dict] = []

                def capture_sampler(*, _calls=sampler_calls, **kwargs):
                    _calls.append(kwargs)
                    return _zero_sampler(**kwargs)

                with patch(
                    "irodori_tts.inference_runtime.sample_euler_rf_cfg",
                    side_effect=capture_sampler,
                ):
                    ordinary_result = ordinary_runtime.synthesize(ordinary_request)
                    prepared_result = prepared_runtime.synthesize(
                        prepared_request,
                        prepared_reference=prepared,
                    )

                torch.testing.assert_close(
                    prepared_result.audio, ordinary_result.audio, rtol=0, atol=0
                )
                self.assertEqual(prepared_result.used_seed, ordinary_result.used_seed)
                self.assertEqual(len(sampler_calls), 2)
                ordinary_call, prepared_call = sampler_calls
                for key in (
                    "cfg_scale_text",
                    "cfg_scale_caption",
                    "cfg_scale_speaker",
                    "use_context_kv_cache",
                    "speaker_kv_scale",
                    "speaker_kv_min_t",
                ):
                    self.assertEqual(prepared_call[key], ordinary_call[key])
                torch.testing.assert_close(
                    prepared_call["speaker_state_override"],
                    ordinary_call["speaker_state_override"],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    prepared_call["speaker_mask_override"],
                    ordinary_call["speaker_mask_override"],
                    rtol=0,
                    atol=0,
                )
                self.assertFalse(prepared_call["speaker_mask_override"].any().item())
                if speaker_kv_scale is not None:
                    self.assertIsNone(prepared_call["speaker_kv_scale"])
                    self.assertTrue(
                        any("ignoring speaker_kv_scale" in msg for msg in prepared_result.messages)
                    )
                if seconds is None:
                    self.assertFalse(
                        ordinary_runtime.model.duration_has_speaker[-1].any().item()
                    )
                    self.assertFalse(
                        prepared_runtime.model.duration_has_speaker[-1].any().item()
                    )
                else:
                    self.assertEqual(ordinary_runtime.model.duration_has_speaker, [])
                    self.assertEqual(prepared_runtime.model.duration_has_speaker, [])

    def test_prepared_no_ref_matches_when_checkpoint_disables_speaker_conditioning(self) -> None:
        ordinary_runtime = _make_runtime()
        prepared_runtime = _make_runtime()
        for runtime in (ordinary_runtime, prepared_runtime):
            runtime.model_cfg.use_speaker_condition_resolved = False
            runtime.model.cfg.use_speaker_condition_resolved = False

        prepared = prepared_runtime.prepare_reference_conditioning(no_ref=True)
        self.assertFalse(prepared.speaker_conditioning_enabled)
        self.assertIsNone(prepared.speaker_state)
        request_kwargs = {
            "text": "speaker disabled",
            "seconds": 0.5,
            "min_seconds": 0.5,
            "max_seconds": 0.5,
            "num_steps": 1,
            "seed": 59,
            "trim_tail": False,
            "cfg_guidance_mode": "joint",
            "speaker_kv_scale": 2.0,
        }
        with patch("irodori_tts.inference_runtime.sample_euler_rf_cfg", _zero_sampler):
            ordinary = ordinary_runtime.synthesize(
                SamplingRequest(no_ref=True, **request_kwargs)
            )
            reused = prepared_runtime.synthesize(
                SamplingRequest(**request_kwargs), prepared_reference=prepared
            )
        torch.testing.assert_close(reused.audio, ordinary.audio, rtol=0, atol=0)

    def test_prepared_no_ref_is_the_complete_source_semantic(self) -> None:
        runtime = _make_runtime()
        prepared = runtime.prepare_reference_conditioning(no_ref=True)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            runtime.synthesize(
                SamplingRequest(text="ambiguous", no_ref=True),
                prepared_reference=prepared,
            )


class InferenceRuntimeCharacterLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(31)
        cls.cfg = _small_model_config(use_duration_predictor=True)
        base_model = TextToLatentRFDiT(cls.cfg).eval()
        torch.nn.init.normal_(base_model.out_proj.weight, std=0.02)
        cls.base_state = {
            name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()
        }
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        cls.adapter_a = root / "adapter-a"
        cls.adapter_b = root / "adapter-b"
        cls.adapter_two_targets = root / "adapter-two-targets"
        cls.adapter_duration = root / "adapter-duration"
        cls._save_adapter(cls.adapter_a, scale=0.10)
        cls._save_adapter(cls.adapter_b, scale=-0.15)
        cls._save_adapter(
            cls.adapter_two_targets,
            scale=0.20,
            target_modules=r"^(speaker_encoder\.in_proj|out_proj)$",
        )
        cls._save_adapter(
            cls.adapter_duration,
            scale=0.12,
            modules_to_save=True,
        )
        cls.ref_latent = torch.randn((1, 6, 2))
        cls.ref_mask = torch.tensor([[True, True, True, True, True, False]])

    @classmethod
    def tearDownClass(cls) -> None:
        clear_cached_runtime()
        cls.temp_dir.cleanup()

    @classmethod
    def _save_adapter(
        cls,
        path: Path,
        *,
        scale: float,
        target_modules: str = r"^speaker_encoder\.in_proj$",
        modules_to_save: bool = False,
    ) -> None:
        model = TextToLatentRFDiT(cls.cfg).eval()
        model.load_state_dict(cls.base_state)
        peft_model = get_peft_model(
            model,
            LoraConfig(
                task_type=None,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                target_modules=target_modules,
                modules_to_save=["duration_predictor"] if modules_to_save else None,
                init_lora_weights=True,
            ),
        )
        with torch.no_grad():
            for name, parameter in peft_model.named_parameters():
                if "lora_A" in name:
                    parameter.fill_(0.25)
                elif "lora_B" in name:
                    parameter.fill_(scale)
                elif modules_to_save and ".modules_to_save." in name:
                    parameter.add_(0.4)
        peft_model.save_pretrained(path)

    @classmethod
    def _clone_config(cls, source: Path, name: str, **updates: object) -> Path:
        destination = Path(cls.temp_dir.name) / name
        shutil.copytree(source, destination)
        config_path = destination / "adapter_config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload.update(updates)
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        return destination

    @classmethod
    def _clone_state(cls, source: Path, name: str, transform) -> Path:
        destination = Path(cls.temp_dir.name) / name
        shutil.copytree(source, destination)
        state_path = destination / "adapter_model.safetensors"
        loaded = load_safetensors_file(state_path, device="cpu")
        state = {key: value.clone() for key, value in loaded.items()}
        transformed = transform(state)
        replacement = destination / "adapter_model.replacement.safetensors"
        save_safetensors_file(transformed, replacement)
        state_path.unlink()
        replacement.rename(state_path)
        return destination

    @classmethod
    def _clone_state_bin(cls, source: Path, name: str, transform) -> Path:
        destination = Path(cls.temp_dir.name) / name
        shutil.copytree(source, destination)
        state_path = destination / "adapter_model.safetensors"
        loaded = load_safetensors_file(state_path, device="cpu")
        state = {key: value.clone() for key, value in loaded.items()}
        state_path.unlink()
        torch.save(transform(state), destination / "adapter_model.bin")
        return destination

    def _runtime(self) -> InferenceRuntime:
        return _make_real_runtime(cfg=self.cfg, state_dict=self.base_state)

    def _prepare(
        self,
        runtime: InferenceRuntime,
        adapter: str | Path | None = None,
    ) -> PreparedReferenceConditioning:
        with patch.object(
            runtime,
            "_load_reference_latent",
            return_value=(self.ref_latent.clone(), self.ref_mask.clone()),
        ):
            return runtime.prepare_reference_conditioning(
                lora_adapter=None if adapter is None else str(adapter)
            )

    def _synthesize_prepared(
        self,
        runtime: InferenceRuntime,
        prepared: PreparedReferenceConditioning,
        adapter: str | Path | None = None,
    ) -> None:
        request = SamplingRequest(
            text="character consistency",
            seconds=0.5,
            min_seconds=0.5,
            max_seconds=0.5,
            num_steps=1,
            seed=37,
            trim_tail=False,
            lora_adapter=None if adapter is None else str(adapter),
        )
        with patch("irodori_tts.inference_runtime.sample_euler_rf_cfg", _zero_sampler):
            runtime.synthesize(request, prepared_reference=prepared)

    def test_fresh_runtime_starts_base_ready(self) -> None:
        runtime = self._runtime()
        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.BASE_READY)
        self.assertIsNone(runtime.character_lora_adapter)

    def test_no_lora_finalize_locks_character(self) -> None:
        runtime = self._runtime()
        runtime.finalize_character()

        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)
        self.assertIsNone(runtime.character_lora_adapter)

    def test_first_valid_lora_transitions_through_loading_and_locks(self) -> None:
        runtime = self._runtime()
        from irodori_tts.lora import apply_preflighted_lora_adapter

        def observed_apply(model, preflight):
            self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.LOADING)
            return apply_preflighted_lora_adapter(model, preflight)

        with patch(
            "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
            side_effect=observed_apply,
        ):
            runtime.finalize_character(str(self.adapter_a))

        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)
        self.assertEqual(runtime.character_lora_adapter, str(self.adapter_a.resolve()))
        self.assertEqual(set(runtime.model.peft_config), {"character"})
        self.assertEqual(runtime.model.active_adapters, ["character"])
        self.assertFalse(runtime.model.training)

    def test_prepare_waits_for_loading_to_finish_under_inference_lock(self) -> None:
        runtime = self._runtime()
        from irodori_tts.lora import apply_preflighted_lora_adapter

        load_started = threading.Event()
        release_load = threading.Event()
        prepare_finished = threading.Event()
        errors: list[BaseException] = []

        def slow_apply(model, preflight):
            load_started.set()
            if not release_load.wait(timeout=5):
                raise TimeoutError("test did not release character load")
            return apply_preflighted_lora_adapter(model, preflight)

        def finalize() -> None:
            try:
                runtime.finalize_character(str(self.adapter_a))
            except BaseException as exc:  # pragma: no cover - thread handoff
                errors.append(exc)

        def prepare() -> None:
            try:
                self._prepare(runtime)
                prepare_finished.set()
            except BaseException as exc:  # pragma: no cover - thread handoff
                errors.append(exc)

        with patch(
            "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
            side_effect=slow_apply,
        ):
            load_thread = threading.Thread(target=finalize)
            load_thread.start()
            self.assertTrue(load_started.wait(timeout=5))
            self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.LOADING)

            prepare_thread = threading.Thread(target=prepare)
            prepare_thread.start()
            self.assertFalse(prepare_finished.wait(timeout=0.1))
            release_load.set()
            load_thread.join(timeout=5)
            prepare_thread.join(timeout=5)

        self.assertFalse(load_thread.is_alive())
        self.assertFalse(prepare_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(prepare_finished.is_set())
        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)

    def test_second_different_lora_is_rejected_before_mutation(self) -> None:
        runtime = self._runtime()
        runtime.finalize_character(str(self.adapter_a))
        model_before = runtime.model

        with patch(
            "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
            side_effect=AssertionError("second adapter must not mutate"),
        ) as loader:
            with self.assertRaisesRegex(RuntimeError, "already locked"):
                runtime.finalize_character(str(self.adapter_b))
            with self.assertRaisesRegex(RuntimeError, "adapter switching"):
                self._prepare(runtime, self.adapter_b)

        loader.assert_not_called()
        self.assertIs(runtime.model, model_before)
        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)

    def test_same_lora_explicit_reload_rejects_but_request_reference_is_no_change(self) -> None:
        runtime = self._runtime()
        prepared = self._prepare(runtime, self.adapter_a)
        model_before = runtime.model

        with self.assertRaisesRegex(RuntimeError, "already locked"):
            runtime.finalize_character(str(self.adapter_a))

        with patch(
            "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
            side_effect=AssertionError("matching request must not reload"),
        ) as loader:
            self._synthesize_prepared(runtime, prepared, self.adapter_a)
            prepared_again = self._prepare(runtime, self.adapter_a)

        loader.assert_not_called()
        self.assertIs(runtime.model, model_before)
        self.assertIs(prepared_again._runtime_owner_token, prepared._runtime_owner_token)

    def test_none_means_no_change_but_explicit_off_after_lock_rejects(self) -> None:
        runtime = self._runtime()
        prepared = self._prepare(runtime, self.adapter_a)

        self._synthesize_prepared(runtime, prepared, None)
        with self.assertRaisesRegex(RuntimeError, "base/off"):
            runtime.prepare_reference_conditioning(lora_adapter="off")
        with self.assertRaisesRegex(RuntimeError, "already locked"):
            runtime.finalize_character(None)

        self.assertEqual(runtime.character_lora_adapter, str(self.adapter_a.resolve()))

    def test_prepare_and_generate_are_allowed_only_after_automatic_lock(self) -> None:
        runtime = self._runtime()
        prepared = self._prepare(runtime)

        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)
        self._synthesize_prepared(runtime, prepared)

    def test_post_mutation_failure_enters_failed_and_rejects_prepare_and_generate(self) -> None:
        runtime = self._runtime()
        with (
            patch(
                "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
                side_effect=RuntimeError("controlled application failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "now FAILED"),
        ):
            runtime.finalize_character(str(self.adapter_a))

        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.FAILED)
        with self.assertRaisesRegex(RuntimeError, "FAILED"):
            runtime.prepare_reference_conditioning(no_ref=True)
        with self.assertRaisesRegex(RuntimeError, "FAILED"):
            runtime.synthesize(SamplingRequest(text="failed", no_ref=True))

    def test_closed_runtime_rejects_prepare_and_generate(self) -> None:
        runtime = self._runtime()
        runtime.finalize_character()
        runtime.unload()

        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.CLOSED)
        with self.assertRaisesRegex(RuntimeError, "CLOSED"):
            runtime.prepare_reference_conditioning(no_ref=True)
        with self.assertRaisesRegex(RuntimeError, "CLOSED"):
            runtime.synthesize(SamplingRequest(text="closed", no_ref=True))

    def test_preflight_failure_preserves_base_ready_and_allows_retry(self) -> None:
        incomplete = self._clone_state(
            self.adapter_a,
            "preflight-incomplete",
            lambda state: {
                key: value for key, value in state.items() if ".lora_B." not in key
            },
        )
        runtime = self._runtime()
        with self.assertRaisesRegex(ValueError, "Incomplete character LoRA"):
            runtime.finalize_character(str(incomplete))

        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.BASE_READY)
        self.assertIsNone(runtime.character_lora_adapter)
        runtime.finalize_character(str(self.adapter_a))
        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)

    def test_first_adapter_payload_must_be_complete(self) -> None:
        cases = {
            "a-only": (
                self.adapter_a,
                lambda state: {
                    key: value for key, value in state.items() if ".lora_B." not in key
                },
            ),
            "b-only": (
                self.adapter_a,
                lambda state: {
                    key: value for key, value in state.items() if ".lora_A." not in key
                },
            ),
            "missing-target-layer": (
                self.adapter_two_targets,
                lambda state: {
                    key: value
                    for key, value in state.items()
                    if "out_proj.lora_" not in key
                },
            ),
            "auxiliary-only": (
                self.adapter_duration,
                lambda state: {
                    key: value for key, value in state.items() if "lora_" not in key
                },
            ),
        }
        for name, (source, transform) in cases.items():
            with self.subTest(name=name):
                adapter = self._clone_state(source, name, transform)
                runtime = self._runtime()
                with patch(
                    "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
                    side_effect=AssertionError("incomplete payload must not mutate"),
                ) as loader:
                    with self.assertRaisesRegex(ValueError, "Incomplete character LoRA"):
                        runtime.finalize_character(str(adapter))
                loader.assert_not_called()
                self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.BASE_READY)

    def test_unknown_and_shared_original_destinations_reject_before_mutation(self) -> None:
        def with_unknown(state):
            state["base_model.model.speaker_encoder.in_proj.lora_unknown.weight"] = torch.ones(
                (16, 8)
            )
            return state

        def with_original(state):
            state[
                "base_model.model.speaker_encoder.in_proj.original_module.weight"
            ] = self.base_state["speaker_encoder.in_proj.weight"].clone()
            return state

        for name, transform in (
            ("unknown-destination", with_unknown),
            ("shared-original", with_original),
        ):
            with self.subTest(name=name):
                adapter = self._clone_state(self.adapter_a, name, transform)
                runtime = self._runtime()
                with patch(
                    "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
                    side_effect=AssertionError("unsafe payload must not mutate"),
                ) as loader:
                    with self.assertRaisesRegex(ValueError, "Unsupported character LoRA"):
                        runtime.finalize_character(str(adapter))
                loader.assert_not_called()
                self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.BASE_READY)

    def test_incompatible_tensor_shape_rejects_before_mutation(self) -> None:
        def wrong_shape(state):
            key = next(key for key in state if ".lora_A." in key)
            state[key] = state[key][:-1].clone()
            return state

        adapter = self._clone_state(self.adapter_a, "wrong-shape", wrong_shape)
        runtime = self._runtime()
        with patch(
            "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
            side_effect=AssertionError("bad shape must not mutate"),
        ) as loader:
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                runtime.finalize_character(str(adapter))
        loader.assert_not_called()
        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.BASE_READY)

    def test_sparse_lora_layouts_reject_before_mutation(self) -> None:
        def replace_lora_a(state, layout):
            key = next(key for key in state if ".lora_A." in key)
            state[key] = layout(state[key])
            return state

        cases = {
            "sparse-coo": lambda tensor: tensor.to_sparse(),
            "sparse-csr": lambda tensor: tensor.to_sparse_csr(),
        }
        for name, layout in cases.items():
            with self.subTest(name=name):
                adapter = self._clone_state_bin(
                    self.adapter_a,
                    name,
                    lambda state, layout=layout: replace_lora_a(state, layout),
                )
                runtime = self._runtime()
                model_before = runtime.model
                with patch(
                    "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
                    side_effect=AssertionError("sparse payload must not mutate"),
                ) as loader:
                    with self.assertRaisesRegex(ValueError, "tensor layout"):
                        runtime.finalize_character(str(adapter))

                loader.assert_not_called()
                self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.BASE_READY)
                self.assertIs(runtime.model, model_before)

    def test_sparse_duration_predictor_layout_rejects_before_mutation(self) -> None:
        def make_auxiliary_sparse(state):
            key = next(
                key
                for key in state
                if ".duration_predictor." in key and "lora_" not in key
            )
            state[key] = state[key].to_sparse()
            return state

        adapter = self._clone_state_bin(
            self.adapter_duration,
            "sparse-duration-predictor",
            make_auxiliary_sparse,
        )
        runtime = self._runtime()
        model_before = runtime.model
        with patch(
            "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
            side_effect=AssertionError("sparse auxiliary payload must not mutate"),
        ) as loader:
            with self.assertRaisesRegex(ValueError, "tensor layout"):
                runtime.finalize_character(str(adapter))

        loader.assert_not_called()
        self.assertIs(runtime.model, model_before)
        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.BASE_READY)

    def test_sparse_layout_rejection_keeps_runtime_exclusive_to_claiming_session(
        self,
    ) -> None:
        def make_lora_a_sparse(state):
            key = next(key for key in state if ".lora_A." in key)
            state[key] = state[key].to_sparse()
            return state

        adapter = self._clone_state_bin(
            self.adapter_a,
            "sparse-claimed-runtime",
            make_lora_a_sparse,
        )
        key = RuntimeKey(checkpoint="sparse-claim", model_device="cpu")
        claimed = self._runtime()
        next_pristine = self._runtime()
        clear_cached_runtime()
        with patch.object(InferenceRuntime, "from_key", side_effect=[claimed, next_pristine]):
            preload_cached_runtime(key)
            claimed_runtime, reused_preload = claim_cached_runtime(key)
            with self.assertRaisesRegex(ValueError, "tensor layout"):
                claimed_runtime.finalize_character(str(adapter))
            next_preload = preload_cached_runtime(key)

        self.assertFalse(reused_preload)
        self.assertTrue(next_preload.reloaded)
        self.assertEqual(claimed.lifecycle_state, RuntimeLifecycle.BASE_READY)
        self.assertEqual(next_pristine.lifecycle_state, RuntimeLifecycle.BASE_READY)
        clear_cached_runtime()
        self.assertEqual(claimed.lifecycle_state, RuntimeLifecycle.BASE_READY)
        self.assertEqual(next_pristine.lifecycle_state, RuntimeLifecycle.CLOSED)

    def test_supported_ordinary_and_duration_predictor_adapters_load(self) -> None:
        for adapter in (self.adapter_a, self.adapter_duration):
            with self.subTest(adapter=adapter.name):
                runtime = self._runtime()
                runtime.finalize_character(str(adapter))
                self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)
                self.assertEqual(set(runtime.model.peft_config), {"character"})

    def test_unsupported_character_peft_surface_rejects_early(self) -> None:
        cases = {
            "dora": {"use_dora": True},
            "rslora": {"use_rslora": True},
            "trainable-tokens": {
                "trainable_token_indices": {"text_encoder.text_embedding": [1, 2]}
            },
            "target-parameters": {
                "target_parameters": ["speaker_encoder.in_proj.weight"]
            },
            "initializer-false": {"init_lora_weights": False},
            "initializer-gaussian": {"init_lora_weights": "gaussian"},
            "bias-all": {"bias": "all"},
            "lora-bias": {"lora_bias": True},
            "rank-pattern": {"rank_pattern": {"speaker_encoder.in_proj": 8}},
            "alpha-pattern": {"alpha_pattern": {"speaker_encoder.in_proj": 16}},
            "wrong-rank": {"r": 8},
            "wrong-alpha": {"lora_alpha": 16},
            "arbitrary-modules-to-save": {"modules_to_save": ["speaker_norm"]},
            "qalora": {"use_qalora": True},
        }
        for name, updates in cases.items():
            with self.subTest(name=name):
                adapter = self._clone_config(self.adapter_a, f"unsupported-{name}", **updates)
                runtime = self._runtime()
                with patch(
                    "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
                    side_effect=AssertionError("unsupported config must not mutate"),
                ) as loader:
                    with self.assertRaisesRegex(ValueError, "Unsupported character LoRA"):
                        runtime.finalize_character(str(adapter))
                loader.assert_not_called()
                self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.BASE_READY)

    def test_noop_peft_application_is_detected_and_runtime_fails(self) -> None:
        runtime = self._runtime()
        with (
            patch(
                "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
                side_effect=lambda model, _preflight: model,
            ),
            self.assertRaisesRegex(RuntimeError, "now FAILED"),
        ):
            runtime.finalize_character(str(self.adapter_a))

        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.FAILED)
        self.assertIn("post-load verification failed", runtime._failure_reason)

    def test_prepared_owner_accepts_same_runtime_and_rejects_another(self) -> None:
        runtime_a = self._runtime()
        runtime_b = self._runtime()
        prepared = self._prepare(runtime_a)

        self._synthesize_prepared(runtime_a, prepared)
        with self.assertRaisesRegex(ValueError, "runtime-owner mismatch"):
            self._synthesize_prepared(runtime_b, prepared)

    def test_closed_runtime_and_new_runtime_reject_old_prepared(self) -> None:
        runtime_a = self._runtime()
        prepared = self._prepare(runtime_a)
        runtime_a.unload()

        with self.assertRaisesRegex(RuntimeError, "CLOSED"):
            self._synthesize_prepared(runtime_a, prepared)

        runtime_b = self._runtime()
        with self.assertRaisesRegex(ValueError, "runtime-owner mismatch"):
            self._synthesize_prepared(runtime_b, prepared)

    def test_same_adapter_path_does_not_make_prepared_cross_runtime_compatible(self) -> None:
        runtime_a = self._runtime()
        runtime_b = self._runtime()
        prepared = self._prepare(runtime_a, self.adapter_a)
        runtime_b.finalize_character(str(self.adapter_a))

        with self.assertRaisesRegex(ValueError, "runtime-owner mismatch"):
            self._synthesize_prepared(runtime_b, prepared, self.adapter_a)

    def test_preload_then_claim_detaches_runtime_from_pristine_cache(self) -> None:
        clear_cached_runtime()
        key = RuntimeKey(checkpoint="test", model_device="cpu")
        first = _make_runtime()
        second = _make_runtime()
        with patch.object(InferenceRuntime, "from_key", side_effect=[first, second]):
            preload = preload_cached_runtime(key)
            claimed_first, created_first = claim_cached_runtime(key)
            claimed_first.finalize_character()
            claimed_second, created_second = claim_cached_runtime(key)

        self.assertTrue(preload.reloaded)
        self.assertFalse(created_first)
        self.assertTrue(created_second)
        self.assertIs(claimed_first, first)
        self.assertIs(claimed_second, second)
        self.assertIsNot(claimed_second, claimed_first)
        clear_cached_runtime()

    def test_locked_failed_and_closed_runtimes_are_not_pristine_cache_hits(self) -> None:
        for terminal in ("locked", "failed", "closed"):
            with self.subTest(terminal=terminal):
                clear_cached_runtime()
                key = RuntimeKey(checkpoint=f"test-{terminal}", model_device="cpu")
                first = self._runtime()
                second = self._runtime()
                with patch.object(InferenceRuntime, "from_key", side_effect=[first, second]):
                    preload_cached_runtime(key)
                    claimed_first, reused_preload = claim_cached_runtime(key)
                    self.assertFalse(reused_preload)
                    if terminal == "failed":
                        with (
                            patch(
                                "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
                                side_effect=RuntimeError("controlled failure"),
                            ),
                            self.assertRaises(RuntimeError),
                        ):
                            claimed_first.finalize_character(str(self.adapter_a))
                    elif terminal == "locked":
                        claimed_first.finalize_character()
                    else:
                        claimed_first.unload()
                    second_preload = preload_cached_runtime(key)
                    claimed_second, created_second = claim_cached_runtime(key)

                self.assertTrue(second_preload.reloaded)
                self.assertFalse(created_second)
                self.assertIs(claimed_second, second)
                self.assertIsNot(claimed_second, claimed_first)
                clear_cached_runtime()

    def test_base_ready_preload_reuses_the_same_pristine_runtime(self) -> None:
        clear_cached_runtime()
        key = RuntimeKey(checkpoint="test-base", model_device="cpu")
        runtime = _make_runtime()
        with patch.object(InferenceRuntime, "from_key", return_value=runtime) as factory:
            first = preload_cached_runtime(key)
            second = preload_cached_runtime(key)

        self.assertTrue(first.reloaded)
        self.assertFalse(second.reloaded)
        factory.assert_called_once_with(key)
        clear_cached_runtime()

    def test_two_concurrent_claims_never_receive_the_same_cached_runtime(self) -> None:
        clear_cached_runtime()
        key = RuntimeKey(checkpoint="test-concurrent-claim", model_device="cpu")
        first = _make_runtime()
        second = _make_runtime()
        barrier = threading.Barrier(3)
        results: list[tuple[InferenceRuntime, bool]] = []
        errors: list[BaseException] = []

        def claim() -> None:
            try:
                barrier.wait(timeout=5)
                results.append(claim_cached_runtime(key))
            except BaseException as exc:  # pragma: no cover - thread handoff
                errors.append(exc)

        with patch.object(InferenceRuntime, "from_key", side_effect=[first, second]) as factory:
            preload_cached_runtime(key)
            threads = [threading.Thread(target=claim) for _ in range(2)]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=5)
            for thread in threads:
                thread.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual({id(runtime) for runtime, _ in results}, {id(first), id(second)})
        self.assertEqual(sorted(reloaded for _, reloaded in results), [False, True])
        self.assertEqual(factory.call_count, 2)
        clear_cached_runtime()

    def test_claimed_runtime_survives_different_key_cache_pressure_during_finalize(
        self,
    ) -> None:
        clear_cached_runtime()
        key_a = RuntimeKey(checkpoint="test-claim-a", model_device="cpu")
        key_b = RuntimeKey(checkpoint="test-claim-b", model_device="cpu")
        claimed = self._runtime()
        replacement = self._runtime()
        load_started = threading.Event()
        release_load = threading.Event()
        errors: list[BaseException] = []
        from irodori_tts.lora import apply_preflighted_lora_adapter

        def slow_apply(model, preflight):
            load_started.set()
            if not release_load.wait(timeout=5):
                raise TimeoutError("test did not release character load")
            return apply_preflighted_lora_adapter(model, preflight)

        def finalize() -> None:
            try:
                claimed.finalize_character(str(self.adapter_a))
            except BaseException as exc:  # pragma: no cover - thread handoff
                errors.append(exc)

        with (
            patch.object(InferenceRuntime, "from_key", side_effect=[claimed, replacement]),
            patch.object(claimed, "unload", wraps=claimed.unload) as claimed_unload,
            patch(
                "irodori_tts.inference_runtime.apply_preflighted_lora_adapter",
                side_effect=slow_apply,
            ),
        ):
            preload_cached_runtime(key_a)
            claimed_runtime, reused_preload = claim_cached_runtime(key_a)
            self.assertIs(claimed_runtime, claimed)
            self.assertFalse(reused_preload)
            finalize_thread = threading.Thread(target=finalize)
            finalize_thread.start()
            self.assertTrue(load_started.wait(timeout=5))

            pressure = preload_cached_runtime(key_b)
            self.assertTrue(pressure.reloaded)
            claimed_unload.assert_not_called()

            release_load.set()
            finalize_thread.join(timeout=5)

        self.assertFalse(finalize_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(claimed.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)
        claimed_unload.assert_not_called()
        clear_cached_runtime()

    def test_cache_eviction_closes_only_cache_owned_pristine_runtime(self) -> None:
        clear_cached_runtime()
        key_a = RuntimeKey(checkpoint="test-evict-a", model_device="cpu")
        key_b = RuntimeKey(checkpoint="test-evict-b", model_device="cpu")
        first = _make_runtime()
        second = _make_runtime()
        with patch.object(InferenceRuntime, "from_key", side_effect=[first, second]):
            preload_cached_runtime(key_a)
            preload_cached_runtime(key_b)

        self.assertEqual(first.lifecycle_state, RuntimeLifecycle.CLOSED)
        self.assertEqual(second.lifecycle_state, RuntimeLifecycle.BASE_READY)
        claimed_second, reused_preload = claim_cached_runtime(key_b)
        self.assertIs(claimed_second, second)
        self.assertFalse(reused_preload)
        clear_cached_runtime()

    def test_character_runtime_session_reuses_and_closes_on_replacement(self) -> None:
        clear_cached_runtime()
        key = RuntimeKey(checkpoint="test-session", model_device="cpu")
        first = _make_runtime()
        second = _make_runtime()
        session = CharacterRuntimeSession()
        with patch(
            "irodori_tts.inference_runtime.claim_cached_runtime",
            side_effect=[(first, True), (second, True)],
        ) as claim:
            acquired_first, first_reloaded = session.acquire(key, lora_adapter=None)
            acquired_again, second_reloaded = session.acquire(key, lora_adapter=None)
            acquired_replacement, replacement_reloaded = session.acquire(
                key,
                lora_adapter="different-character",
            )

        self.assertIs(acquired_first, first)
        self.assertIs(acquired_again, first)
        self.assertIs(acquired_replacement, second)
        self.assertTrue(first_reloaded)
        self.assertFalse(second_reloaded)
        self.assertTrue(replacement_reloaded)
        self.assertEqual(claim.call_count, 2)
        self.assertEqual(first.lifecycle_state, RuntimeLifecycle.CLOSED)
        session.close()
        self.assertEqual(second.lifecycle_state, RuntimeLifecycle.CLOSED)

    def test_ref_embed_preparation_bypasses_speaker_encoder(self) -> None:
        runtime = self._runtime()
        embedding = torch.randn((3, self.cfg.speaker_dim))
        path = Path(self.temp_dir.name) / "reference.speaker.safetensors"
        save_speaker_inversion_safetensors(path, {"speaker_embedding": embedding})

        with patch.object(
            runtime.model.speaker_encoder,
            "forward",
            wraps=runtime.model.speaker_encoder.forward,
        ) as speaker_forward:
            prepared = runtime.prepare_reference_conditioning(ref_embed=str(path))

        self.assertEqual(speaker_forward.call_count, 0)
        torch.testing.assert_close(prepared.speaker_state, embedding.unsqueeze(0), rtol=0, atol=0)
        torch.testing.assert_close(
            prepared.speaker_mask,
            torch.ones((1, 3), dtype=torch.bool),
        )

    def test_ordinary_wav_synthesize_under_locked_lora_encodes_reference_once(self) -> None:
        runtime = self._runtime()
        request = SamplingRequest(
            text="ordinary wav lora",
            ref_wav="reference.wav",
            ref_normalize_db=None,
            seconds=None,
            min_seconds=0.5,
            max_seconds=1.0,
            num_steps=1,
            seed=41,
            trim_tail=False,
            lora_adapter=str(self.adapter_a),
        )

        with (
            patch(
                "irodori_tts.inference_runtime._load_audio",
                return_value=(torch.zeros((1, 8)), 4),
            ),
            patch("irodori_tts.inference_runtime.sample_euler_rf_cfg", _zero_sampler),
            patch.object(
                runtime.model.speaker_encoder,
                "forward",
                wraps=runtime.model.speaker_encoder.forward,
            ) as speaker_forward,
        ):
            result = runtime.synthesize(request)

        self.assertEqual(runtime.lifecycle_state, RuntimeLifecycle.CHARACTER_LOCKED)
        self.assertEqual(runtime.codec.encode_count, 1)
        self.assertEqual(speaker_forward.call_count, 1)
        self.assertEqual(result.used_seed, 41)
        self.assertEqual(result.audio.shape, (1, 4))



if __name__ == "__main__":
    unittest.main()
