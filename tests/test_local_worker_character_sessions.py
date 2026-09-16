from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch

from irodori_tts import local_worker as local_worker_module
from irodori_tts.inference_runtime import PreparedReferenceConditioning
from irodori_tts.local_worker import CAPABILITIES, PROTOCOL_VERSION, LocalWorker
from irodori_tts.voice_engine import VoiceGenerationResult


class _CharacterSessionFakeEngine:
    instances: list[_CharacterSessionFakeEngine] = []
    next_generation = 1
    fail_claim = False

    def __init__(self, reference_audio: Path | None, output_dir: Path) -> None:
        self.reference_audio = reference_audio
        self.output_dir = output_dir
        self.runtime_generation = f"{type(self).next_generation:032x}"
        type(self).next_generation += 1
        self.character_state = "base_ready"
        self.claim_calls: list[Path | None] = []
        self.prepare_calls = 0
        self.generate_calls = 0
        self.no_ref_calls = 0
        self.close_calls = 0
        type(self).instances.append(self)

    def load(self) -> None:
        return None

    def claim_character(self, lora_adapter: Path | None = None) -> None:
        self.claim_calls.append(lora_adapter)
        if type(self).fail_claim:
            raise RuntimeError("injected claim failure")
        if lora_adapter is not None and not (
            (lora_adapter / "adapter_config.json").is_file()
            and (lora_adapter / "adapter_model.safetensors").is_file()
        ):
            raise ValueError("invalid adapter")
        self.character_state = "character_locked"

    def prepare_reference_conditioning(self, snapshot: bytes, **_preprocessing):
        self.prepare_calls += 1
        value = float(max(1, sum(snapshot) % 127))
        return PreparedReferenceConditioning(
            speaker_state=torch.full((1, 2, 2), value),
            speaker_mask=torch.ones((1, 2), dtype=torch.bool),
            _lora_adapter=None,
        )

    def generate_with_prepared(
        self,
        _text,
        _prepared_reference,
        settings=None,
        output_path=None,
    ) -> VoiceGenerationResult:
        self.generate_calls += 1
        return VoiceGenerationResult(Path(output_path), settings.seed, 0.25)

    def generate_no_ref(
        self,
        _text,
        settings=None,
        output_path=None,
    ) -> VoiceGenerationResult:
        self.no_ref_calls += 1
        return VoiceGenerationResult(Path(output_path), settings.seed, 0.25)

    def close(self) -> None:
        self.close_calls += 1


class _Harness:
    def __init__(self) -> None:
        self.output = io.StringIO()
        self.error = io.StringIO()
        self.worker = LocalWorker(
            output_stream=self.output,
            error_stream=self.error,
            engine_factory=_CharacterSessionFakeEngine,
        )

    def request(self, payload: dict) -> dict:
        self.worker.process_line(json.dumps(payload))
        return json.loads(self.output.getvalue().splitlines()[-1])

    def responses(self) -> list[dict]:
        return [json.loads(line) for line in self.output.getvalue().splitlines()]


class LocalWorkerCharacterSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        _CharacterSessionFakeEngine.instances = []
        _CharacterSessionFakeEngine.next_generation = 1
        _CharacterSessionFakeEngine.fail_claim = False

    @staticmethod
    def _preload(output_dir: Path, *, explicit: bool = True, request_id: str = "preload"):
        request = {
            "id": request_id,
            "type": "preload",
            "output_dir": str(output_dir),
        }
        if explicit:
            request["character_session_mode"] = "explicit"
        return request

    @staticmethod
    def _base_claim(generation: str, session_id: str = "character-1", request_id="claim"):
        return {
            "id": request_id,
            "type": "claim_character",
            "runtime_generation": generation,
            "character_session_id": session_id,
            "character_voice": {"kind": "base"},
        }

    @staticmethod
    def _lora_claim(
        generation: str,
        path: Path | str,
        *,
        session_id: str = "character-1",
        request_id: str = "claim",
        backend: str = "irodori",
        base_model_id: str = "Aratako/Irodori-TTS-500M-v3",
    ) -> dict:
        return {
            "id": request_id,
            "type": "claim_character",
            "runtime_generation": generation,
            "character_session_id": session_id,
            "character_voice": {
                "kind": "lora",
                "path": str(path),
                "compatibility": {
                    "backend": backend,
                    "base_model_id": base_model_id,
                },
            },
        }

    @staticmethod
    def _make_adapter(path: Path, *, valid: bool = True) -> Path:
        path.mkdir()
        if valid:
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (path / "adapter_model.safetensors").write_bytes(b"adapter")
        return path

    @staticmethod
    def _reference(path: Path) -> dict:
        return {"files": [str(path)]}

    def test_protocol_v4_legacy_and_explicit_preload_modes_are_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            legacy = _Harness()
            legacy_preload = legacy.request(self._preload(base / "legacy", explicit=False))
            legacy_repeat = legacy.request(
                self._preload(base / "legacy", explicit=False, request_id="preload-2")
            )
            legacy_switch = legacy.request(
                self._preload(base / "legacy", explicit=True, request_id="preload-3")
            )

            explicit = _Harness()
            explicit_preload = explicit.request(self._preload(base / "explicit"))
            explicit_repeat = explicit.request(
                self._preload(base / "explicit", request_id="preload-2")
            )
            explicit_switch = explicit.request(
                self._preload(base / "explicit", explicit=False, request_id="preload-3")
            )

            null_mode = _Harness().request(
                {
                    **self._preload(base / "null", explicit=False),
                    "character_session_mode": None,
                }
            )

        self.assertEqual(PROTOCOL_VERSION, 4)
        self.assertEqual(
            list(CAPABILITIES),
            ["no_ref", "prepared_voice_handles", "character_sessions"],
        )
        self.assertEqual(legacy_preload["character_session_mode"], "legacy")
        self.assertEqual(legacy_repeat["runtime_generation"], legacy_preload["runtime_generation"])
        self.assertTrue(legacy_repeat["already_loaded"])
        self.assertEqual(legacy_switch["error"]["reason"], "character_session_mode_conflict")
        self.assertEqual(explicit_preload["character_session_mode"], "explicit")
        self.assertEqual(explicit_preload["character_state"], "unclaimed")
        self.assertEqual(
            explicit_repeat["runtime_generation"], explicit_preload["runtime_generation"]
        )
        self.assertTrue(explicit_repeat["already_loaded"])
        self.assertEqual(explicit_switch["error"]["reason"], "character_session_mode_conflict")
        self.assertEqual(null_mode["error"]["reason"], "invalid_character_session_mode")
        self.assertEqual(len(_CharacterSessionFakeEngine.instances), 2)

    def test_base_claim_is_stable_idempotent_and_preload_does_not_reset_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            harness = _Harness()
            output_dir = Path(temp_dir) / "outputs"
            preload = harness.request(self._preload(output_dir))
            generation = preload["runtime_generation"]
            first = harness.request(self._base_claim(generation))
            retry = harness.request(self._base_claim(generation, request_id="claim-retry"))
            after_lock = harness.request(self._preload(output_dir, request_id="preload-after-lock"))

        engine = _CharacterSessionFakeEngine.instances[0]
        self.assertEqual(first["runtime_generation"], generation)
        self.assertEqual(first["character_session_id"], "character-1")
        self.assertEqual(first["character_state"], "character_locked")
        self.assertEqual(first["voice_kind"], "base")
        self.assertFalse(first["already_claimed"])
        self.assertTrue(retry["already_claimed"])
        self.assertEqual(engine.claim_calls, [None])
        self.assertEqual(after_lock["runtime_generation"], generation)
        self.assertEqual(after_lock["character_state"], "character_locked")
        self.assertTrue(after_lock["already_loaded"])

    def test_base_claim_conflicts_reject_voice_and_session_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            adapter = self._make_adapter(base / "adapter")
            harness = _Harness()
            generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]
            harness.request(self._base_claim(generation))
            different_session = harness.request(
                self._base_claim(generation, session_id="character-2", request_id="other-session")
            )
            base_to_lora = harness.request(
                self._lora_claim(generation, adapter, request_id="base-to-lora")
            )

        self.assertEqual(different_session["error"]["code"], "character_claim_conflict")
        self.assertEqual(different_session["error"]["reason"], "different_character_session")
        self.assertEqual(base_to_lora["error"]["reason"], "different_character_voice")
        self.assertEqual(_CharacterSessionFakeEngine.instances[0].claim_calls, [None])

    def test_lora_claim_identity_is_idempotent_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            adapter_a = self._make_adapter(base / "adapter-a")
            adapter_b = self._make_adapter(base / "adapter-b")
            harness = _Harness()
            generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]

            success = harness.request(self._lora_claim(generation, adapter_a))
            retry = harness.request(self._lora_claim(generation, adapter_a, request_id="retry"))
            switch = harness.request(self._lora_claim(generation, adapter_b, request_id="switch"))
            metadata_change = harness.request(
                self._lora_claim(
                    generation,
                    adapter_a,
                    backend="other",
                    request_id="metadata-change",
                )
            )
            to_base = harness.request(self._base_claim(generation, request_id="to-base"))
            other_session = harness.request(
                self._lora_claim(
                    generation,
                    adapter_a,
                    session_id="character-2",
                    request_id="other-session",
                )
            )

        engine = _CharacterSessionFakeEngine.instances[0]
        self.assertEqual(success["voice_kind"], "lora")
        self.assertTrue(retry["already_claimed"])
        self.assertEqual(switch["error"]["code"], "character_claim_conflict")
        self.assertEqual(metadata_change["error"]["reason"], "different_character_voice")
        self.assertEqual(to_base["error"]["code"], "character_claim_conflict")
        self.assertEqual(other_session["error"]["reason"], "different_character_session")
        self.assertEqual(engine.claim_calls, [adapter_a.resolve()])
        self.assertEqual(engine.prepare_calls, 0)

    def test_successful_lora_replay_does_not_revalidate_moved_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            adapter = self._make_adapter(base / "adapter")
            harness = _Harness()
            generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]
            with mock.patch.object(
                local_worker_module,
                "_validate_character_voice_resources",
                wraps=local_worker_module._validate_character_voice_resources,
            ) as validate_resources:
                first = harness.request(self._lora_claim(generation, adapter))
                adapter.rename(base / "adapter-moved")
                retry = harness.request(
                    self._lora_claim(
                        generation,
                        adapter.parent / "missing-parent" / ".." / adapter.name,
                        request_id="retry",
                    )
                )

        engine = _CharacterSessionFakeEngine.instances[0]
        self.assertTrue(first["ok"])
        self.assertFalse(first["already_claimed"])
        self.assertTrue(retry["ok"])
        self.assertTrue(retry["already_claimed"])
        self.assertEqual(validate_resources.call_count, 1)
        self.assertEqual(engine.claim_calls, [adapter.resolve()])

    def test_compatibility_failures_poison_runtime(self) -> None:
        cases = (
            ({"backend": "other"}, "compatibility_backend_mismatch", "character-2"),
            (
                {"base_model_id": "Aratako/Irodori-TTS-500M-v4.1"},
                "compatibility_base_model_mismatch",
                "character-1",
            ),
        )
        for overrides, expected_reason, retry_session in cases:
            with self.subTest(reason=expected_reason), tempfile.TemporaryDirectory() as temp_dir:
                base = Path(temp_dir)
                adapter = self._make_adapter(base / "adapter")
                harness = _Harness()
                generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]
                failed = harness.request(
                    self._lora_claim(generation, adapter, request_id="failed", **overrides)
                )
                retry = harness.request(
                    self._base_claim(
                        generation,
                        session_id=retry_session,
                        request_id="retry",
                    )
                )

                self.assertEqual(failed["error"]["reason"], expected_reason)
                self.assertEqual(retry["error"]["code"], "runtime_unavailable")
                self.assertEqual(retry["error"]["reason"], "previous_claim_failed")
                self.assertTrue(harness.worker._claim_failed)
                self.assertIsNone(harness.worker._character_claim)
                self.assertEqual(_CharacterSessionFakeEngine.instances[-1].claim_calls, [])

    def test_missing_and_non_directory_lora_paths_poison_runtime(self) -> None:
        for path_kind in ("missing", "file"):
            with self.subTest(path_kind=path_kind), tempfile.TemporaryDirectory() as temp_dir:
                base = Path(temp_dir)
                adapter = base / "adapter"
                if path_kind == "file":
                    adapter.write_bytes(b"not a directory")
                harness = _Harness()
                generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]
                failed = harness.request(self._lora_claim(generation, adapter))
                retry = harness.request(
                    self._base_claim(
                        generation,
                        session_id="character-2",
                        request_id="retry",
                    )
                )

                self.assertEqual(failed["error"]["reason"], "lora_path_not_directory")
                self.assertEqual(retry["error"]["reason"], "previous_claim_failed")
                self.assertTrue(harness.worker._claim_failed)
                self.assertIsNone(harness.worker._character_claim)
                self.assertEqual(_CharacterSessionFakeEngine.instances[-1].claim_calls, [])

    def test_strict_adapter_validation_failure_poisons_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            invalid_adapter = self._make_adapter(base / "invalid", valid=False)
            harness = _Harness()
            generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]
            failed = harness.request(self._lora_claim(generation, invalid_adapter))
            retry = harness.request(self._base_claim(generation, request_id="retry"))

        engine = _CharacterSessionFakeEngine.instances[0]
        self.assertEqual(failed["error"]["reason"], "character_finalize_failed")
        self.assertEqual(retry["error"]["reason"], "previous_claim_failed")
        self.assertEqual(engine.claim_calls, [invalid_adapter.resolve()])
        self.assertEqual(engine.character_state, "base_ready")
        self.assertTrue(harness.worker._claim_failed)
        self.assertIsNone(harness.worker._character_claim)

    def test_runtime_finalize_failure_poisons_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            adapter = self._make_adapter(base / "adapter")
            harness = _Harness()
            generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]
            _CharacterSessionFakeEngine.fail_claim = True
            failed = harness.request(self._lora_claim(generation, adapter))
            retry = harness.request(self._base_claim(generation, request_id="retry"))

        self.assertEqual(failed["error"]["reason"], "character_finalize_failed")
        self.assertEqual(retry["error"]["reason"], "previous_claim_failed")
        self.assertTrue(harness.worker._claim_failed)
        self.assertIsNone(harness.worker._character_claim)

    def test_failed_claim_blocks_inference_and_allows_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            adapter = self._make_adapter(base / "adapter")
            reference = base / "reference.wav"
            reference.write_bytes(b"reference")
            harness = _Harness()
            generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]
            failed = harness.request(
                self._lora_claim(generation, adapter, backend="other", request_id="failed")
            )
            requests = [
                {
                    "id": "prepare",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "character_session_id": "character-1",
                    "reference": self._reference(reference),
                },
                {
                    "id": "reference",
                    "type": "generate",
                    "text": "hello",
                    "reference": self._reference(reference),
                    "runtime_generation": generation,
                    "character_session_id": "character-1",
                    "output_path": str(base / "reference-out.wav"),
                },
                {
                    "id": "handle",
                    "type": "generate",
                    "text": "hello",
                    "prepared_voice_id": "0" * 32,
                    "runtime_generation": generation,
                    "character_session_id": "character-1",
                    "output_path": str(base / "handle.wav"),
                },
                {
                    "id": "no-ref",
                    "type": "generate",
                    "text": "hello",
                    "no_ref": True,
                    "runtime_generation": generation,
                    "character_session_id": "character-1",
                    "output_path": str(base / "no-ref.wav"),
                },
            ]
            responses = [harness.request(request) for request in requests]
            shutdown = harness.request({"id": "shutdown", "type": "shutdown"})

        engine = _CharacterSessionFakeEngine.instances[0]
        self.assertEqual(failed["error"]["reason"], "compatibility_backend_mismatch")
        self.assertTrue(
            all(response["error"]["code"] == "runtime_unavailable" for response in responses)
        )
        self.assertTrue(
            all(response["error"]["reason"] == "previous_claim_failed" for response in responses)
        )
        self.assertEqual(engine.prepare_calls, 0)
        self.assertEqual(engine.generate_calls, 0)
        self.assertEqual(engine.no_ref_calls, 0)
        self.assertTrue(shutdown["ok"])
        self.assertEqual(engine.close_calls, 1)

    def test_stale_and_malformed_claim_requests_do_not_poison_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            harness = _Harness()
            generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]
            stale = harness.request(
                self._lora_claim("f" * 32, base / "missing", request_id="stale")
            )
            harness.worker.process_line("{")
            malformed_json = harness.responses()[-1]
            invalid_id = harness.request(
                self._lora_claim(generation, base / "missing", request_id="")
            )
            malformed_voice = harness.request(
                {
                    "id": "malformed-voice",
                    "type": "claim_character",
                    "runtime_generation": generation,
                    "character_session_id": "character-1",
                    "character_voice": {"kind": "lora"},
                }
            )
            self.assertFalse(harness.worker._claim_failed)
            success = harness.request(self._base_claim(generation))

        self.assertEqual(stale["error"]["reason"], "runtime_generation_mismatch")
        self.assertEqual(malformed_json["error"]["code"], "invalid_json")
        self.assertEqual(invalid_id["error"]["code"], "invalid_request")
        self.assertEqual(malformed_voice["error"]["reason"], "invalid_lora_voice_fields")
        self.assertTrue(success["ok"])
        self.assertFalse(success["already_claimed"])

    def test_explicit_mode_gates_all_inference_forms_before_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "reference.wav"
            reference.write_bytes(b"reference")
            harness = _Harness()
            generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]
            requests = [
                {
                    "id": "prepare",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "reference": self._reference(reference),
                },
                {
                    "id": "reference",
                    "type": "generate",
                    "text": "hello",
                    "reference_audio": str(reference),
                    "output_path": str(base / "reference.wav.out"),
                },
                {
                    "id": "handle",
                    "type": "generate",
                    "text": "hello",
                    "prepared_voice_id": "0" * 32,
                    "runtime_generation": generation,
                    "output_path": str(base / "handle.wav"),
                },
                {
                    "id": "no-ref",
                    "type": "generate",
                    "text": "hello",
                    "no_ref": True,
                    "output_path": str(base / "no-ref.wav"),
                },
            ]
            responses = [harness.request(request) for request in requests]

        self.assertTrue(
            all(response["error"]["code"] == "character_not_locked" for response in responses)
        )
        engine = _CharacterSessionFakeEngine.instances[0]
        self.assertEqual(engine.prepare_calls, 0)
        self.assertEqual(engine.generate_calls, 0)
        self.assertEqual(engine.no_ref_calls, 0)

    def test_post_claim_identity_validation_and_prepared_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "reference.wav"
            reference.write_bytes(b"reference")
            harness = _Harness()
            generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]
            harness.request(self._base_claim(generation))

            prepare_request = {
                "id": "prepare",
                "type": "prepare_voice",
                "runtime_generation": generation,
                "character_session_id": "character-1",
                "reference": self._reference(reference),
            }
            missing_session = harness.request(
                {
                    key: value
                    for key, value in prepare_request.items()
                    if key != "character_session_id"
                }
            )
            wrong_session = harness.request(
                {**prepare_request, "id": "wrong-session", "character_session_id": "character-2"}
            )
            wrong_generation = harness.request(
                {**prepare_request, "id": "wrong-generation", "runtime_generation": "f" * 32}
            )
            missing_generate_session = harness.request(
                {
                    "id": "generate-missing-session",
                    "type": "generate",
                    "text": "hello",
                    "no_ref": True,
                    "runtime_generation": generation,
                    "output_path": str(base / "missing-session.wav"),
                }
            )
            wrong_generate_session = harness.request(
                {
                    "id": "generate-wrong-session",
                    "type": "generate",
                    "text": "hello",
                    "no_ref": True,
                    "runtime_generation": generation,
                    "character_session_id": "character-2",
                    "output_path": str(base / "wrong-session.wav"),
                }
            )
            prepared = harness.request(prepare_request)
            cache_hit = harness.request({**prepare_request, "id": "prepare-hit"})
            reference_first = harness.request(
                {
                    "id": "generate-reference",
                    "type": "generate",
                    "text": "hello",
                    "reference": self._reference(reference),
                    "runtime_generation": generation,
                    "character_session_id": "character-1",
                    "output_path": str(base / "reference.wav.out"),
                }
            )
            generated = harness.request(
                {
                    "id": "generate-handle",
                    "type": "generate",
                    "text": "hello",
                    "prepared_voice_id": prepared["prepared_voice_id"],
                    "runtime_generation": generation,
                    "character_session_id": "character-1",
                    "output_path": str(base / "handle.wav"),
                }
            )
            fallback = harness.request(
                {
                    "id": "generate-fallback",
                    "type": "generate",
                    "text": "hello",
                    "prepared_voice_id": "f" * 32,
                    "runtime_generation": generation,
                    "character_session_id": "character-1",
                    "reference": self._reference(reference),
                    "reference_revision": prepared["reference_revision"],
                    "output_path": str(base / "fallback.wav"),
                }
            )
            no_ref = harness.request(
                {
                    "id": "generate-no-ref",
                    "type": "generate",
                    "text": "hello",
                    "no_ref": True,
                    "runtime_generation": generation,
                    "character_session_id": "character-1",
                    "output_path": str(base / "no-ref.wav"),
                }
            )

        self.assertEqual(missing_session["error"]["reason"], "missing_character_session_id")
        self.assertEqual(wrong_session["error"]["code"], "character_session_mismatch")
        self.assertEqual(wrong_generation["error"]["reason"], "runtime_generation_mismatch")
        self.assertEqual(
            missing_generate_session["error"]["reason"], "missing_character_session_id"
        )
        self.assertEqual(wrong_generate_session["error"]["code"], "character_session_mismatch")
        self.assertTrue(prepared["ok"])
        self.assertTrue(cache_hit["cache_hit"])
        self.assertEqual(cache_hit["prepared_voice_id"], prepared["prepared_voice_id"])
        self.assertTrue(reference_first["ok"])
        self.assertTrue(generated["ok"])
        self.assertTrue(generated["cache_hit"])
        self.assertTrue(fallback["ok"])
        self.assertTrue(fallback["cache_hit"])
        self.assertTrue(no_ref["ok"])
        engine = _CharacterSessionFakeEngine.instances[0]
        self.assertEqual(engine.prepare_calls, 1)
        self.assertEqual(engine.generate_calls, 3)
        self.assertEqual(engine.no_ref_calls, 1)

    def test_duplicate_claims_are_serialized_and_shutdown_releases_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            harness = _Harness()
            generation = harness.request(self._preload(base / "outputs"))["runtime_generation"]
            barrier = threading.Barrier(3)

            def claim(request_id: str) -> None:
                barrier.wait()
                harness.worker.process_line(
                    json.dumps(self._base_claim(generation, request_id=request_id))
                )

            threads = [
                threading.Thread(target=claim, args=("claim-1",)),
                threading.Thread(target=claim, args=("claim-2",)),
            ]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join()

            claim_responses = [
                response for response in harness.responses() if response["id"].startswith("claim-")
            ]
            shutdown = harness.request({"id": "shutdown", "type": "shutdown"})
            after_shutdown = harness.request(
                self._base_claim(generation, request_id="after-shutdown")
            )

        engine = _CharacterSessionFakeEngine.instances[0]
        self.assertEqual(engine.claim_calls, [None])
        self.assertEqual(
            {response["already_claimed"] for response in claim_responses}, {False, True}
        )
        self.assertTrue(shutdown["ok"])
        self.assertEqual(after_shutdown["error"]["reason"], "worker_closed")
        self.assertEqual(engine.close_calls, 1)
        self.assertIsNone(harness.worker.engine)
        self.assertIsNone(harness.worker.cache)
        self.assertIsNone(harness.worker._character_claim)


if __name__ == "__main__":
    unittest.main()
