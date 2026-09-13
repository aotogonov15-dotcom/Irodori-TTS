from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from voice_engine import VoiceEngine, VoiceGenerationResult, VoiceGenerationSettings


class FakeRuntime:
    def __init__(self) -> None:
        self.requests = []

    def synthesize(self, request, log_fn=None):
        self.requests.append(request)
        return SimpleNamespace(
            audio=b"dummy audio",
            sample_rate=24000,
            used_seed=1234,
            total_to_decode=0.123,
        )


class VoiceEngineLoadTest(unittest.TestCase):
    def test_load_without_reference_initializes_global_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            output_dir = base_dir / "outputs"
            runtime = object()

            with (
                patch(
                    "irodori_tts.voice_engine.hf_hub_download",
                    return_value=str(base_dir / "model.safetensors"),
                ),
                patch(
                    "irodori_tts.voice_engine.InferenceRuntime.from_key",
                    return_value=runtime,
                ) as from_key,
            ):
                engine = VoiceEngine(None, output_dir)
                engine.load()

            self.assertTrue(output_dir.is_dir())
            self.assertIs(engine._runtime, runtime)
            self.assertTrue(engine.is_loaded)
            self.assertEqual(from_key.call_count, 1)

    def test_load_without_reference_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            with (
                patch(
                    "irodori_tts.voice_engine.hf_hub_download",
                    return_value=str(base_dir / "model.safetensors"),
                ) as download,
                patch(
                    "irodori_tts.voice_engine.InferenceRuntime.from_key",
                    return_value=object(),
                ) as from_key,
            ):
                engine = VoiceEngine(None, base_dir / "outputs")
                engine.load()
                engine.load()

            self.assertEqual(download.call_count, 1)
            self.assertEqual(from_key.call_count, 1)

    def test_generate_requires_reference_after_reference_less_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = VoiceEngine(None, Path(temp_dir) / "outputs")
            engine._runtime = FakeRuntime()

            with self.assertRaisesRegex(FileNotFoundError, "参照音声"):
                engine.generate("こんにちは")

    def test_generate_accepts_explicit_reference_after_reference_less_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            reference = base_dir / "reference.wav"
            reference.write_bytes(b"dummy wav")
            fake_runtime = FakeRuntime()
            engine = VoiceEngine(None, base_dir / "outputs")
            engine._runtime = fake_runtime

            with patch(
                "irodori_tts.voice_engine.save_wav",
                return_value=base_dir / "generated.wav",
            ):
                engine.generate("こんにちは", reference_audio=reference)

            self.assertEqual(fake_runtime.requests[0].ref_wav, str(reference))
            self.assertIsNone(engine.reference_audio)

    def test_load_failure_keeps_engine_unloaded_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            with (
                patch(
                    "irodori_tts.voice_engine.hf_hub_download",
                    return_value=str(base_dir / "model.safetensors"),
                ),
                patch(
                    "irodori_tts.voice_engine.InferenceRuntime.from_key",
                    side_effect=RuntimeError("codec load failed"),
                ),
            ):
                engine = VoiceEngine(None, base_dir / "outputs")
                with self.assertRaisesRegex(RuntimeError, "codec load failed"):
                    engine.load()

            self.assertFalse(engine.is_loaded)


class VoiceEngineGenerateTest(unittest.TestCase):
    def test_default_settings_match_existing_sampling_values(self) -> None:
        settings = VoiceGenerationSettings()

        self.assertEqual(settings.num_steps, 16)
        self.assertEqual(settings.t_schedule_mode, "sway")
        self.assertEqual(settings.sway_coeff, -1.0)
        self.assertEqual(settings.cfg_guidance_mode, "independent")
        self.assertEqual(settings.cfg_scale_text, 3.0)
        self.assertEqual(settings.cfg_scale_speaker, 7.0)
        self.assertEqual(settings.duration_scale, 1.0)
        self.assertEqual(settings.seed, 1234)
        self.assertEqual(settings.num_candidates, 1)
        self.assertEqual(settings.decode_mode, "sequential")
        self.assertEqual(settings.ref_normalize_db, -16.0)
        self.assertIs(settings.ref_ensure_max, True)
        self.assertEqual(settings.max_ref_seconds, 30.0)

    def test_generate_passes_default_settings_to_sampling_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            fake_runtime = FakeRuntime()
            engine._runtime = fake_runtime

            with patch(
                "irodori_tts.voice_engine.save_wav",
                return_value=base_dir / "generated.wav",
            ):
                engine.generate("こんにちは")

            self._assert_sampling_settings(
                fake_runtime.requests[0],
                VoiceGenerationSettings(),
            )

    def test_generate_passes_custom_settings_to_sampling_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            fake_runtime = FakeRuntime()
            engine._runtime = fake_runtime
            settings = VoiceGenerationSettings(
                num_steps=8,
                t_schedule_mode="linear",
                sway_coeff=0.25,
                cfg_guidance_mode="alternating",
                cfg_scale_text=2.0,
                cfg_scale_speaker=4.5,
                duration_scale=1.25,
                seed=4321,
                num_candidates=2,
                decode_mode="batch",
                ref_normalize_db=-12.0,
                ref_ensure_max=False,
                max_ref_seconds=12.5,
            )

            with patch(
                "irodori_tts.voice_engine.save_wav",
                return_value=base_dir / "generated.wav",
            ):
                engine.generate("こんにちは", settings=settings)

            self._assert_sampling_settings(fake_runtime.requests[0], settings)

    def test_generate_does_not_mutate_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            engine._runtime = FakeRuntime()
            settings = VoiceGenerationSettings(seed=4321, num_steps=8)
            original_settings = VoiceGenerationSettings(seed=4321, num_steps=8)

            with patch(
                "irodori_tts.voice_engine.save_wav",
                return_value=base_dir / "generated.wav",
            ):
                engine.generate("こんにちは", settings=settings)

            self.assertEqual(settings, original_settings)

    def test_generate_uses_specified_reference_audio_without_mutating_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            override_audio = self._write_audio(base_dir / "override.wav")
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            fake_runtime = FakeRuntime()
            engine._runtime = fake_runtime

            with patch(
                "irodori_tts.voice_engine.save_wav",
                return_value=base_dir / "generated.wav",
            ):
                engine.generate("こんにちは", reference_audio=override_audio)

            self.assertEqual(fake_runtime.requests[0].ref_wav, str(override_audio))
            self.assertEqual(engine.reference_audio, default_audio)

    def test_generate_keeps_reference_audio_override_with_custom_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            override_audio = self._write_audio(base_dir / "override.wav")
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            fake_runtime = FakeRuntime()
            engine._runtime = fake_runtime

            with patch(
                "irodori_tts.voice_engine.save_wav",
                return_value=base_dir / "generated.wav",
            ):
                engine.generate(
                    "こんにちは",
                    reference_audio=override_audio,
                    settings=VoiceGenerationSettings(seed=4321),
                )

            request = fake_runtime.requests[0]
            self.assertEqual(request.ref_wav, str(override_audio))
            self.assertEqual(request.seed, 4321)

    def test_generate_uses_default_reference_audio_when_not_specified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            fake_runtime = FakeRuntime()
            engine._runtime = fake_runtime

            with patch(
                "irodori_tts.voice_engine.save_wav",
                return_value=base_dir / "generated.wav",
            ):
                engine.generate("こんにちは")

            self.assertEqual(fake_runtime.requests[0].ref_wav, str(default_audio))

    def test_generate_rejects_missing_specified_reference_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            missing_audio = base_dir / "missing.wav"
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            engine._runtime = FakeRuntime()

            with self.assertRaisesRegex(FileNotFoundError, "参照音声"):
                engine.generate("こんにちは", reference_audio=missing_audio)

            self.assertEqual(engine.reference_audio, default_audio)

    def test_generate_returns_voice_generation_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            saved_path = base_dir / "generated.wav"
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            engine._runtime = FakeRuntime()

            with patch("irodori_tts.voice_engine.save_wav", return_value=saved_path):
                result = engine.generate("こんにちは")

            self.assertEqual(result.output_path, saved_path.resolve())
            self.assertEqual(result.used_seed, 1234)
            self.assertEqual(result.generation_seconds, 0.123)

    def test_generate_uses_custom_output_path_when_specified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            custom_output = base_dir / "nested" / "custom.wav"
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            engine._runtime = FakeRuntime()

            with patch("irodori_tts.voice_engine.save_wav", return_value=custom_output) as save:
                result = engine.generate("こんにちは", output_path=custom_output)

            self.assertEqual(save.call_args.args[0], custom_output)
            self.assertEqual(result.output_path, custom_output.resolve())

    def test_generate_keeps_timestamp_output_path_when_not_specified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_audio = self._write_audio(base_dir / "default.wav")
            engine = VoiceEngine(default_audio, base_dir / "outputs")
            engine._runtime = FakeRuntime()

            with patch(
                "irodori_tts.voice_engine.save_wav",
                return_value=base_dir / "generated.wav",
            ) as save:
                engine.generate("こんにちは")

            self.assertEqual(save.call_args.args[0].parent, base_dir / "outputs")
            self.assertTrue(save.call_args.args[0].name.startswith("voice_"))
            self.assertEqual(save.call_args.args[0].suffix, ".wav")

    def test_root_voice_engine_compatibility_imports_package_api(self) -> None:
        from irodori_tts.voice_engine import VoiceEngine as PackageVoiceEngine
        from irodori_tts.voice_engine import VoiceGenerationResult as PackageResult
        from irodori_tts.voice_engine import VoiceGenerationSettings as PackageSettings

        self.assertIs(VoiceEngine, PackageVoiceEngine)
        self.assertIs(VoiceGenerationSettings, PackageSettings)
        self.assertIs(VoiceGenerationResult, PackageResult)

    def _write_audio(self, path: Path) -> Path:
        path.write_bytes(b"dummy wav")
        return path

    def _assert_sampling_settings(
        self,
        request,
        settings: VoiceGenerationSettings,
    ) -> None:
        self.assertEqual(request.num_steps, settings.num_steps)
        self.assertEqual(request.t_schedule_mode, settings.t_schedule_mode)
        self.assertEqual(request.sway_coeff, settings.sway_coeff)
        self.assertEqual(request.cfg_guidance_mode, settings.cfg_guidance_mode)
        self.assertEqual(request.cfg_scale_text, settings.cfg_scale_text)
        self.assertEqual(request.cfg_scale_speaker, settings.cfg_scale_speaker)
        self.assertEqual(request.duration_scale, settings.duration_scale)
        self.assertEqual(request.seed, settings.seed)
        self.assertEqual(request.num_candidates, settings.num_candidates)
        self.assertEqual(request.decode_mode, settings.decode_mode)
        self.assertEqual(request.ref_normalize_db, settings.ref_normalize_db)
        self.assertEqual(request.ref_ensure_max, settings.ref_ensure_max)
        self.assertEqual(request.max_ref_seconds, settings.max_ref_seconds)


if __name__ == "__main__":
    unittest.main()
