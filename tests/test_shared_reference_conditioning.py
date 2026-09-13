from __future__ import annotations

import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model

from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    PreparedReferenceConditioning,
    SamplingRequest,
)
from irodori_tts.lora import apply_lora, lora_adapter_mutates_base_parameters
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
    runtime._effective_conditioning_state_revision = 0
    runtime._lora_load_failure = None
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
    runtime._lora_adapter_names = {}
    runtime._effective_conditioning_state_revision = 0
    runtime._lora_load_failure = None
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


class InferenceRuntimeLoraConditioningTest(unittest.TestCase):
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
        cls.adapter_a = Path(cls.temp_dir.name) / "adapter-a"
        cls.adapter_b = Path(cls.temp_dir.name) / "adapter-b"
        cls.adapter_bias_all = Path(cls.temp_dir.name) / "adapter-bias-all"
        cls.adapter_bias_lora_only = Path(cls.temp_dir.name) / "adapter-bias-lora-only"
        cls.adapter_modules_to_save = Path(cls.temp_dir.name) / "adapter-modules-to-save"
        cls.adapter_pissa = Path(cls.temp_dir.name) / "adapter-pissa"
        cls.adapter_olora = Path(cls.temp_dir.name) / "adapter-olora"
        cls._save_adapter(cls.adapter_a, scale=0.10)
        cls._save_adapter(cls.adapter_b, scale=-0.15)
        cls._save_adapter(
            cls.adapter_bias_all,
            scale=0.20,
            bias="all",
            bias_shift=0.75,
        )
        cls._save_adapter(
            cls.adapter_bias_lora_only,
            scale=-0.20,
            bias="lora_only",
            bias_shift=-0.50,
        )
        cls._save_adapter(
            cls.adapter_modules_to_save,
            scale=0.12,
            modules_to_save="speaker_norm",
            modules_to_save_shift=0.40,
        )
        cls._save_adapter(cls.adapter_pissa, scale=0.14, init_lora_weights="pissa")
        cls._save_adapter(cls.adapter_olora, scale=-0.18, init_lora_weights="olora")
        cls.ref_latent = torch.randn((1, 6, 2))
        cls.ref_mask = torch.tensor([[True, True, True, True, True, False]])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    @classmethod
    def _save_adapter(
        cls,
        path: Path,
        *,
        scale: float,
        bias: str = "none",
        bias_shift: float = 0.0,
        modules_to_save: str = "none",
        modules_to_save_shift: float = 0.0,
        init_lora_weights: bool | str = True,
    ) -> None:
        model = TextToLatentRFDiT(cls.cfg).eval()
        model.load_state_dict(cls.base_state)
        config = {
            "lora_enabled": True,
            "lora_r": 2,
            "lora_alpha": 2,
            "lora_dropout": 0.0,
            "lora_bias": bias,
            "lora_target_modules": r"^speaker_encoder\.in_proj$",
            "lora_modules_to_save": modules_to_save,
        }
        if init_lora_weights is True:
            peft_model = apply_lora(model, config)
        else:
            peft_model = get_peft_model(
                model,
                LoraConfig(
                    task_type=None,
                    inference_mode=False,
                    r=2,
                    lora_alpha=2,
                    lora_dropout=0.0,
                    bias=bias,
                    target_modules=r"^speaker_encoder\.in_proj$",
                    modules_to_save=(
                        None if modules_to_save == "none" else [modules_to_save]
                    ),
                    init_lora_weights=init_lora_weights,
                ),
            )
        touched = 0
        touched_bias = 0
        touched_modules_to_save = 0
        with torch.no_grad():
            for name, parameter in peft_model.named_parameters():
                if "lora_A" in name:
                    parameter.fill_(0.25)
                    touched += 1
                elif "lora_B" in name:
                    parameter.fill_(scale)
                    touched += 1
                elif bias_shift and "bias" in name and parameter.requires_grad:
                    parameter.add_(
                        torch.arange(
                            parameter.numel(),
                            dtype=parameter.dtype,
                            device=parameter.device,
                        ).reshape_as(parameter)
                        * bias_shift
                    )
                    touched_bias += 1
                elif modules_to_save_shift and ".modules_to_save." in name:
                    parameter.add_(modules_to_save_shift)
                    touched_modules_to_save += 1
        if touched == 0:
            raise AssertionError("test adapter did not target the speaker encoder")
        if bias_shift and touched_bias == 0:
            raise AssertionError("test adapter did not expose trainable bias parameters")
        if modules_to_save_shift and touched_modules_to_save == 0:
            raise AssertionError("test adapter did not wrap modules_to_save")
        peft_model.save_pretrained(path)

    def _runtime(self) -> InferenceRuntime:
        return _make_real_runtime(cfg=self.cfg, state_dict=self.base_state)

    def _prepare(
        self,
        runtime: InferenceRuntime,
        adapter: str | Path | None,
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
        adapter: str | Path | None,
    ) -> None:
        request = SamplingRequest(
            text="adapter consistency",
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

    def test_base_prepare_to_base_synthesize_passes(self) -> None:
        runtime = self._runtime()
        prepared = self._prepare(runtime, None)

        self.assertIsNone(prepared._lora_adapter)
        self._synthesize_prepared(runtime, prepared, None)

    def test_same_lora_prepare_and_synthesize_passes(self) -> None:
        runtime = self._runtime()
        prepared = self._prepare(runtime, self.adapter_a)

        self.assertEqual(prepared._lora_adapter, str(self.adapter_a.resolve()))
        self._synthesize_prepared(runtime, prepared, self.adapter_a)

    def test_mismatched_prepared_and_request_adapters_are_rejected(self) -> None:
        runtime = self._runtime()
        prepared_base = self._prepare(runtime, None)
        prepared_a = self._prepare(runtime, self.adapter_a)

        mismatch_cases = (
            (prepared_a, None),
            (prepared_base, self.adapter_a),
            (prepared_a, self.adapter_b),
        )
        for prepared, adapter in mismatch_cases:
            with self.subTest(prepared=prepared._lora_adapter, adapter=adapter):
                with self.assertRaisesRegex(ValueError, "LoRA mismatch"):
                    self._synthesize_prepared(runtime, prepared, adapter)

    def test_explicit_base_prepare_ignores_prior_ambient_adapter(self) -> None:
        runtime = self._runtime()
        prepared_a = self._prepare(runtime, self.adapter_a)
        prepared_base = self._prepare(runtime, None)
        oracle_model = TextToLatentRFDiT(self.cfg).eval()
        oracle_model.load_state_dict(self.base_state)

        with torch.inference_mode():
            expected_base, _ = _legacy_speaker_condition_oracle(
                oracle_model, self.ref_latent, self.ref_mask
            )

        self.assertGreater(
            (prepared_a.speaker_state - prepared_base.speaker_state).abs().max().item(),
            1e-6,
        )
        torch.testing.assert_close(prepared_base.speaker_state, expected_base, rtol=0, atol=0)
        self.assertIsNone(prepared_base._lora_adapter)

    def test_bias_adapters_advance_effective_state_and_reject_stale_base(self) -> None:
        for adapter in (self.adapter_bias_all, self.adapter_bias_lora_only):
            with self.subTest(adapter=adapter.name):
                runtime = self._runtime()
                base_before = self._prepare(runtime, None)
                adapter_prepared = self._prepare(runtime, adapter)
                base_after = self._prepare(runtime, None)

                self.assertEqual(base_before._effective_conditioning_state_revision, 0)
                self.assertEqual(adapter_prepared._effective_conditioning_state_revision, 1)
                self.assertEqual(base_after._effective_conditioning_state_revision, 1)
                if adapter == self.adapter_bias_all:
                    self.assertGreater(
                        (base_before.speaker_state - base_after.speaker_state)
                        .abs()
                        .max()
                        .item(),
                        1e-6,
                    )
                with self.assertRaisesRegex(ValueError, "model-state mismatch"):
                    self._synthesize_prepared(runtime, base_before, None)
                self._synthesize_prepared(runtime, base_after, None)

    def test_modules_to_save_restores_base_without_advancing_effective_state(self) -> None:
        runtime = self._runtime()
        base_before = self._prepare(runtime, None)
        adapter_prepared = self._prepare(runtime, self.adapter_modules_to_save)
        base_after = self._prepare(runtime, None)

        self.assertEqual(runtime._effective_conditioning_state_revision, 0)
        self.assertGreater(
            (adapter_prepared.speaker_state - base_before.speaker_state).abs().max().item(),
            1e-6,
        )
        torch.testing.assert_close(
            base_after.speaker_state, base_before.speaker_state, rtol=0, atol=0
        )
        self._synthesize_prepared(runtime, base_before, None)

    def test_dynamic_preflight_covers_peft_initialization_surface(self) -> None:
        cases = (
            ("default", None, True),
            ("true", True, True),
            ("false", False, True),
            ("gaussian", "gaussian", True),
            ("eva", "eva", True),
            ("orthogonal", "orthogonal", True),
            ("pissa", "pissa", False),
            ("pissa_niter", "pissa_niter_4", False),
            ("olora", "olora", False),
            ("corda", "corda", False),
            ("loftq", "loftq", False),
        )
        for name, initialization, is_safe in cases:
            with self.subTest(initialization=name):
                adapter = Path(self.temp_dir.name) / f"preflight-{name}"
                adapter.mkdir()
                payload = {"bias": "none"}
                if initialization is not None:
                    payload["init_lora_weights"] = initialization
                (adapter / "adapter_config.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                if is_safe:
                    self.assertFalse(lora_adapter_mutates_base_parameters(adapter))
                else:
                    with self.assertRaisesRegex(
                        ValueError,
                        "can persistently modify shared base parameters",
                    ):
                        lora_adapter_mutates_base_parameters(adapter)

    def test_base_mutating_initializations_are_rejected_before_runtime_mutation(self) -> None:
        for adapter in (self.adapter_pissa, self.adapter_olora):
            with self.subTest(adapter=adapter.name):
                runtime = self._runtime()
                base_before = self._prepare(runtime, None)
                model_before = runtime.model

                with (
                    patch(
                        "irodori_tts.inference_runtime.load_lora_adapter",
                        side_effect=AssertionError("unsafe adapter must not reach PEFT loading"),
                    ) as loader,
                    self.assertRaisesRegex(
                        ValueError,
                        "can persistently modify shared base parameters",
                    ),
                ):
                    self._prepare(runtime, adapter)

                loader.assert_not_called()
                self.assertIs(runtime.model, model_before)
                self.assertEqual(runtime._effective_conditioning_state_revision, 0)
                self.assertIsNone(runtime._lora_load_failure)
                self.assertEqual(runtime._lora_adapter_names, {})
                self._synthesize_prepared(runtime, base_before, None)
                base_after = self._prepare(runtime, None)
                torch.testing.assert_close(
                    base_after.speaker_state,
                    base_before.speaker_state,
                    rtol=0,
                    atol=0,
                )

    def test_base_prepare_failure_restores_prior_adapter_and_next_request_is_consistent(self) -> None:
        runtime = self._runtime()
        prepared_a = self._prepare(runtime, self.adapter_a)
        adapter_name = runtime._lora_adapter_names[prepared_a._lora_adapter]

        with (
            patch.object(
                runtime,
                "_prepare_reference_conditioning_for_request",
                side_effect=RuntimeError("expected preparation failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "expected preparation failure"),
        ):
            runtime.prepare_reference_conditioning(lora_adapter=None)

        self.assertEqual(runtime.model.active_adapter, adapter_name)
        prepared_base = self._prepare(runtime, None)
        self.assertIsNone(prepared_base._lora_adapter)
        self._synthesize_prepared(runtime, prepared_base, None)

    def test_lora_prepare_failure_cannot_contaminate_next_explicit_base_request(self) -> None:
        runtime = self._runtime()
        oracle_model = TextToLatentRFDiT(self.cfg).eval()
        oracle_model.load_state_dict(self.base_state)
        with torch.inference_mode():
            expected_base, _ = _legacy_speaker_condition_oracle(
                oracle_model, self.ref_latent, self.ref_mask
            )

        with (
            patch.object(
                runtime,
                "_prepare_reference_conditioning_for_request",
                side_effect=RuntimeError("expected LoRA preparation failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "expected LoRA preparation failure"),
        ):
            runtime.prepare_reference_conditioning(lora_adapter=str(self.adapter_a))

        prepared_base = self._prepare(runtime, None)
        torch.testing.assert_close(prepared_base.speaker_state, expected_base, rtol=0, atol=0)
        self._synthesize_prepared(runtime, prepared_base, None)

    def test_bias_conditioning_failure_invalidates_pre_mutation_base(self) -> None:
        runtime = self._runtime()
        prepared_base = self._prepare(runtime, None)

        with (
            patch.object(
                runtime,
                "_prepare_reference_conditioning_for_request",
                side_effect=RuntimeError("expected bias conditioning failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "expected bias conditioning failure"),
        ):
            runtime.prepare_reference_conditioning(lora_adapter=str(self.adapter_bias_all))

        self.assertEqual(runtime._effective_conditioning_state_revision, 1)
        with self.assertRaisesRegex(ValueError, "model-state mismatch"):
            self._synthesize_prepared(runtime, prepared_base, None)
        prepared_base_after = self._prepare(runtime, None)
        self._synthesize_prepared(runtime, prepared_base_after, None)

    def test_adapter_load_failure_marks_runtime_unavailable_until_reload(self) -> None:
        runtime = self._runtime()
        prepared_base = self._prepare(runtime, None)

        with (
            patch(
                "irodori_tts.inference_runtime.load_lora_adapter",
                side_effect=RuntimeError("expected adapter load failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "may have partially changed"),
        ):
            self._prepare(runtime, self.adapter_a)

        with self.assertRaisesRegex(RuntimeError, "unavailable after a failed dynamic LoRA load"):
            self._synthesize_prepared(runtime, prepared_base, None)

        replacement_runtime = self._runtime()
        replacement_prepared = self._prepare(replacement_runtime, None)
        self._synthesize_prepared(replacement_runtime, replacement_prepared, None)

    def test_adapter_resolve_failure_leaves_effective_state_unchanged(self) -> None:
        runtime = self._runtime()
        prepared_base = self._prepare(runtime, None)
        missing = Path(self.temp_dir.name) / "missing-adapter"

        with self.assertRaises(FileNotFoundError):
            runtime.prepare_reference_conditioning(lora_adapter=str(missing))

        self.assertEqual(runtime._effective_conditioning_state_revision, 0)
        self.assertIsNone(runtime._lora_load_failure)
        self._synthesize_prepared(runtime, prepared_base, None)

    def test_synthesis_failure_after_bias_load_invalidates_old_base(self) -> None:
        runtime = self._runtime()
        prepared_base = self._prepare(runtime, None)
        request = SamplingRequest(
            text="bias synthesis failure",
            no_ref=True,
            seconds=0.5,
            min_seconds=0.5,
            max_seconds=0.5,
            num_steps=1,
            seed=61,
            trim_tail=False,
            lora_adapter=str(self.adapter_bias_all),
        )
        with (
            patch(
                "irodori_tts.inference_runtime.sample_euler_rf_cfg",
                side_effect=RuntimeError("expected synthesis failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "expected synthesis failure"),
        ):
            runtime.synthesize(request)

        with self.assertRaisesRegex(ValueError, "model-state mismatch"):
            self._synthesize_prepared(runtime, prepared_base, None)
        prepared_base_after = self._prepare(runtime, None)
        self._synthesize_prepared(runtime, prepared_base_after, None)

    def test_base_lora_base_lora_base_sequence_is_explicit(self) -> None:
        runtime = self._runtime()
        base_first = self._prepare(runtime, None)
        prepared_a = self._prepare(runtime, self.adapter_a)
        base_middle = self._prepare(runtime, None)
        prepared_b = self._prepare(runtime, self.adapter_b)
        base_last = self._prepare(runtime, None)

        torch.testing.assert_close(base_middle.speaker_state, base_first.speaker_state, rtol=0, atol=0)
        torch.testing.assert_close(base_last.speaker_state, base_first.speaker_state, rtol=0, atol=0)
        self.assertGreater(
            (prepared_a.speaker_state - prepared_b.speaker_state).abs().max().item(),
            1e-6,
        )

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
        torch.testing.assert_close(prepared.speaker_mask, torch.ones((1, 3), dtype=torch.bool))

    def test_ordinary_wav_synthesize_under_lora_uses_one_consistent_adapter(self) -> None:
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

        with patch(
            "irodori_tts.inference_runtime._load_audio",
            return_value=(torch.zeros((1, 8)), 4),
        ):
            result = runtime.synthesize(request)

        self.assertEqual(runtime.codec.encode_count, 1)
        self.assertEqual(result.used_seed, 41)
        self.assertEqual(result.audio.shape, (1, 4))

    def test_ordinary_wav_bias_lora_is_consistent_and_invalidates_old_base(self) -> None:
        runtime = self._runtime()
        prepared_base = self._prepare(runtime, None)
        request = SamplingRequest(
            text="ordinary wav bias lora",
            ref_wav="reference.wav",
            ref_normalize_db=None,
            seconds=None,
            min_seconds=0.5,
            max_seconds=1.0,
            num_steps=1,
            seed=73,
            trim_tail=False,
            lora_adapter=str(self.adapter_bias_all),
        )

        with patch(
            "irodori_tts.inference_runtime._load_audio",
            return_value=(torch.zeros((1, 8)), 4),
        ):
            result = runtime.synthesize(request)

        self.assertEqual(runtime.codec.encode_count, 1)
        self.assertEqual(runtime._effective_conditioning_state_revision, 1)
        self.assertEqual(result.used_seed, 73)
        self.assertEqual(result.audio.shape, (1, 4))
        with self.assertRaisesRegex(ValueError, "model-state mismatch"):
            self._synthesize_prepared(runtime, prepared_base, None)


if __name__ == "__main__":
    unittest.main()
