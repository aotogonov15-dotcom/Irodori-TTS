from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from irodori_tts.inference_runtime import PreparedReferenceConditioning
from irodori_tts.local_worker import LocalWorker, run_worker
from irodori_tts.prepared_voice_cache import (
    ReferencePreprocessing,
    snapshot_reference_file,
)
from voice_engine import VoiceGenerationResult


class _PreparedFakeEngine:
    instances: list[_PreparedFakeEngine] = []
    next_generation = 1

    def __init__(self, reference_audio, output_dir) -> None:
        self.reference_audio = reference_audio
        self.output_dir = output_dir
        self.runtime_generation = f"{type(self).next_generation:032x}"
        type(self).next_generation += 1
        self.prepare_calls: list[bytes] = []
        self.generate_calls: list[dict] = []
        self.no_ref_calls = 0
        self.load_count = 0
        self.close_count = 0
        type(self).instances.append(self)

    def load(self) -> None:
        self.load_count += 1

    def prepare_reference_conditioning(self, snapshot, **preprocessing):
        self.prepare_calls.append(snapshot)
        value = float(sum(snapshot) % 127)
        return PreparedReferenceConditioning(
            speaker_state=torch.full((1, 2, 2), value),
            speaker_mask=torch.ones((1, 2), dtype=torch.bool),
            _lora_adapter=None,
        )

    def generate_with_prepared(
        self,
        text,
        prepared_reference,
        settings=None,
        output_path=None,
    ) -> VoiceGenerationResult:
        self.generate_calls.append(
            {
                "text": text,
                "prepared_reference": prepared_reference,
                "settings": settings,
                "output_path": output_path,
            }
        )
        if text == "fail":
            raise RuntimeError("post-prepare synthesize failed")
        return VoiceGenerationResult(Path(output_path), settings.seed, 0.5)

    def generate_no_ref(
        self,
        text,
        settings=None,
        output_path=None,
    ) -> VoiceGenerationResult:
        self.no_ref_calls += 1
        return VoiceGenerationResult(Path(output_path), settings.seed, 0.25)

    def close(self) -> None:
        self.close_count += 1


class _Harness:
    def __init__(self, **worker_options) -> None:
        self.output = io.StringIO()
        self.error = io.StringIO()
        self.worker = LocalWorker(
            output_stream=self.output,
            error_stream=self.error,
            engine_factory=_PreparedFakeEngine,
            **worker_options,
        )

    def request(self, payload: dict) -> dict:
        self.worker.process_line(json.dumps(payload))
        return json.loads(self.output.getvalue().splitlines()[-1])

    def preload(self, output_dir: Path) -> dict:
        return self.request(
            {
                "id": "preload",
                "type": "preload",
                "output_dir": str(output_dir),
            }
        )


class LocalWorkerPreparedVoiceProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        _PreparedFakeEngine.instances = []
        _PreparedFakeEngine.next_generation = 1

    @staticmethod
    def _reference(path: Path, preprocessing: dict | None = None) -> dict:
        payload = {"files": [str(path)]}
        if preprocessing is not None:
            payload["preprocessing"] = preprocessing
        return payload

    @staticmethod
    def _generate(
        output: Path,
        *,
        text: str = "hello",
        **conditioning,
    ) -> dict:
        return {
            "id": "generate",
            "type": "generate",
            "text": text,
            "output_path": str(output),
            **conditioning,
        }

    def test_prepare_voice_miss_then_hit_returns_same_opaque_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"audio-one")
            harness = _Harness()
            preload = harness.preload(base / "outputs")
            request = {
                "id": "prepare-1",
                "type": "prepare_voice",
                "runtime_generation": preload["runtime_generation"],
                "reference": self._reference(reference),
            }

            miss = harness.request(request)
            hit = harness.request({**request, "id": "prepare-2"})

        self.assertTrue(miss["ok"])
        self.assertFalse(miss["cache_hit"])
        self.assertTrue(hit["cache_hit"])
        self.assertEqual(miss["prepared_voice_id"], hit["prepared_voice_id"])
        self.assertRegex(miss["prepared_voice_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(len(_PreparedFakeEngine.instances[0].prepare_calls), 1)

    def test_same_path_replaced_and_preprocessing_changes_create_new_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"first")
            harness = _Harness()
            generation = harness.preload(base / "outputs")["runtime_generation"]

            first = harness.request(
                {
                    "id": "prepare-1",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "reference": self._reference(reference),
                }
            )
            reference.write_bytes(b"second")
            replaced = harness.request(
                {
                    "id": "prepare-2",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "reference": self._reference(reference),
                }
            )
            changed_options = harness.request(
                {
                    "id": "prepare-3",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "reference": self._reference(
                        reference,
                        {"ref_normalize_db": -12.0},
                    ),
                }
            )

        self.assertNotEqual(first["prepared_voice_id"], replaced["prepared_voice_id"])
        self.assertNotEqual(replaced["prepared_voice_id"], changed_options["prepared_voice_id"])

    def test_reference_miss_prepares_once_and_first_utterance_uses_same_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"one-shot")
            harness = _Harness()
            generation = harness.preload(base / "outputs")["runtime_generation"]

            response = harness.request(
                self._generate(
                    base / "reply.wav",
                    runtime_generation=generation,
                    reference=self._reference(reference),
                )
            )

        engine = _PreparedFakeEngine.instances[0]
        self.assertTrue(response["ok"])
        self.assertFalse(response["cache_hit"])
        self.assertTrue(response["cached"])
        self.assertEqual(len(engine.prepare_calls), 1)
        self.assertEqual(len(engine.generate_calls), 1)
        self.assertIs(
            engine.generate_calls[0]["prepared_reference"],
            harness.worker.cache.acquire_handle(response["prepared_voice_id"]).prepared,
        )

    def test_generate_with_handle_does_not_reread_or_reprepare_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"resident")
            harness = _Harness()
            generation = harness.preload(base / "outputs")["runtime_generation"]
            prepared = harness.request(
                {
                    "id": "prepare",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "reference": self._reference(reference),
                }
            )
            reference.unlink()

            generated = harness.request(
                self._generate(
                    base / "reply.wav",
                    runtime_generation=generation,
                    prepared_voice_id=prepared["prepared_voice_id"],
                    reference=self._reference(reference),
                    reference_revision=prepared["reference_revision"],
                )
            )

        self.assertTrue(generated["ok"])
        self.assertTrue(generated["cache_hit"])
        self.assertEqual(len(_PreparedFakeEngine.instances[0].prepare_calls), 1)

    def test_stale_handle_fallback_prepares_and_synthesizes_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"fallback")
            revision = snapshot_reference_file(
                reference,
                ReferencePreprocessing(),
            ).revision
            harness = _Harness()
            generation = harness.preload(base / "outputs")["runtime_generation"]

            response = harness.request(
                self._generate(
                    base / "reply.wav",
                    runtime_generation=generation,
                    prepared_voice_id="f" * 32,
                    reference=self._reference(reference),
                    reference_revision=revision,
                )
            )

        engine = _PreparedFakeEngine.instances[0]
        self.assertTrue(response["ok"])
        self.assertIsNotNone(response["prepared_voice_id"])
        self.assertNotEqual(response["prepared_voice_id"], "f" * 32)
        self.assertEqual(len(engine.prepare_calls), 1)
        self.assertEqual(len(engine.generate_calls), 1)
        self.assertIn("event=fallback", harness.error.getvalue())

    def test_changed_fallback_content_rejects_before_prepare_or_synthesize(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"before")
            expected = snapshot_reference_file(
                reference,
                ReferencePreprocessing(),
            ).revision
            reference.write_bytes(b"after")
            harness = _Harness()
            generation = harness.preload(base / "outputs")["runtime_generation"]

            response = harness.request(
                self._generate(
                    base / "reply.wav",
                    runtime_generation=generation,
                    prepared_voice_id="f" * 32,
                    reference=self._reference(reference),
                    reference_revision=expected,
                )
            )

        engine = _PreparedFakeEngine.instances[0]
        self.assertEqual(response["error"]["code"], "invalid_reference")
        self.assertEqual(response["error"]["reason"], "revision_mismatch")
        self.assertEqual(engine.prepare_calls, [])
        self.assertEqual(engine.generate_calls, [])

    def test_malformed_stale_and_generation_mismatch_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            harness = _Harness()
            generation = harness.preload(base / "outputs")["runtime_generation"]

            malformed = harness.request(
                self._generate(
                    base / "bad.wav",
                    runtime_generation=generation,
                    prepared_voice_id="not-a-handle",
                )
            )
            stale = harness.request(
                self._generate(
                    base / "stale.wav",
                    runtime_generation=generation,
                    prepared_voice_id="f" * 32,
                )
            )
            mismatch = harness.request(
                self._generate(
                    base / "old.wav",
                    runtime_generation="old-runtime",
                    prepared_voice_id="f" * 32,
                )
            )

        self.assertEqual(malformed["error"]["code"], "invalid_handle")
        self.assertEqual(stale["error"]["code"], "stale_handle")
        self.assertEqual(mismatch["error"]["code"], "runtime_unavailable")
        self.assertEqual(mismatch["error"]["reason"], "runtime_generation_mismatch")

    def test_no_ref_returns_no_handle_and_rejects_contradictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"voice")
            harness = _Harness()
            generation = harness.preload(base / "outputs")["runtime_generation"]

            generated = harness.request(
                self._generate(
                    base / "no-ref.wav",
                    runtime_generation=generation,
                    no_ref=True,
                )
            )
            contradictory = harness.request(
                self._generate(
                    base / "bad.wav",
                    no_ref=True,
                    reference=self._reference(reference),
                )
            )
            prepare = harness.request(
                {
                    "id": "prepare",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "no_ref": True,
                }
            )

        self.assertTrue(generated["ok"])
        self.assertIsNone(generated["prepared_voice_id"])
        self.assertFalse(generated["cached"])
        self.assertEqual(_PreparedFakeEngine.instances[0].no_ref_calls, 1)
        self.assertEqual(contradictory["error"]["code"], "invalid_request")
        self.assertEqual(prepare["error"]["code"], "invalid_request")

    def test_oversized_explicit_prepare_fails_but_generate_uses_it_once_uncached(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"oversized")
            harness = _Harness(cache_max_bytes=1)
            generation = harness.preload(base / "outputs")["runtime_generation"]
            reference_payload = self._reference(reference)

            explicit = harness.request(
                {
                    "id": "prepare",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "reference": reference_payload,
                }
            )
            engine = _PreparedFakeEngine.instances[0]
            engine.prepare_calls.clear()
            generated = harness.request(
                self._generate(
                    base / "reply.wav",
                    runtime_generation=generation,
                    reference=reference_payload,
                )
            )

        self.assertEqual(explicit["error"]["code"], "prepare_failed")
        self.assertEqual(explicit["error"]["reason"], "capacity_exceeded")
        self.assertTrue(generated["ok"])
        self.assertFalse(generated["cached"])
        self.assertIsNone(generated["prepared_voice_id"])
        self.assertEqual(len(engine.prepare_calls), 1)
        self.assertEqual(len(engine.generate_calls), 1)

    def test_evicted_handle_is_stale_and_reprepare_gets_new_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            first_path = base / "first.wav"
            second_path = base / "second.wav"
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")
            harness = _Harness(cache_max_entries=1)
            generation = harness.preload(base / "outputs")["runtime_generation"]

            first = harness.request(
                {
                    "id": "first",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "reference": self._reference(first_path),
                }
            )
            harness.request(
                {
                    "id": "second",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "reference": self._reference(second_path),
                }
            )
            stale = harness.request(
                self._generate(
                    base / "stale.wav",
                    runtime_generation=generation,
                    prepared_voice_id=first["prepared_voice_id"],
                )
            )
            replacement = harness.request(
                {
                    "id": "replacement",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "reference": self._reference(first_path),
                }
            )

        self.assertEqual(stale["error"]["code"], "stale_handle")
        self.assertNotEqual(first["prepared_voice_id"], replacement["prepared_voice_id"])

    def test_generation_failure_does_not_retry_synthesize(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"voice")
            harness = _Harness()
            generation = harness.preload(base / "outputs")["runtime_generation"]

            response = harness.request(
                self._generate(
                    base / "reply.wav",
                    text="fail",
                    runtime_generation=generation,
                    reference=self._reference(reference),
                )
            )

        engine = _PreparedFakeEngine.instances[0]
        self.assertEqual(response["error"]["code"], "generation_failed")
        self.assertEqual(len(engine.prepare_calls), 1)
        self.assertEqual(len(engine.generate_calls), 1)

    def test_shutdown_clears_cache_before_engine_close_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"voice")
            harness = _Harness()
            generation = harness.preload(base / "outputs")["runtime_generation"]
            harness.request(
                {
                    "id": "prepare",
                    "type": "prepare_voice",
                    "runtime_generation": generation,
                    "reference": self._reference(reference),
                }
            )

            response = harness.request({"id": "shutdown", "type": "shutdown"})
            harness.worker.close()

        stderr = harness.error.getvalue()
        self.assertTrue(response["ok"])
        self.assertLess(stderr.index("event=cache_close"), stderr.index("event=engine_close"))
        self.assertEqual(_PreparedFakeEngine.instances[0].close_count, 1)

    def test_worker_restart_uses_new_generation_and_eof_style_close_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"voice")
            first = _Harness()
            first_generation = first.preload(base / "one")["runtime_generation"]
            old_prepared = first.request(
                {
                    "id": "prepare",
                    "type": "prepare_voice",
                    "runtime_generation": first_generation,
                    "reference": self._reference(reference),
                }
            )
            first.worker.close()
            second = _Harness()
            second_generation = second.preload(base / "two")["runtime_generation"]
            stale_in_current = second.request(
                self._generate(
                    base / "stale.wav",
                    runtime_generation=second_generation,
                    prepared_voice_id=old_prepared["prepared_voice_id"],
                )
            )
            explicit_old_generation = second.request(
                self._generate(
                    base / "old-generation.wav",
                    runtime_generation=first_generation,
                    prepared_voice_id=old_prepared["prepared_voice_id"],
                )
            )

        self.assertNotEqual(first_generation, second_generation)
        self.assertIsNone(first.worker.cache)
        self.assertEqual(_PreparedFakeEngine.instances[0].close_count, 1)
        self.assertEqual(stale_in_current["error"]["code"], "stale_handle")
        self.assertEqual(explicit_old_generation["error"]["code"], "runtime_unavailable")

    def test_protocol_output_failure_still_closes_engine(self) -> None:
        class _BrokenOutput(io.StringIO):
            def write(self, value):
                raise OSError("controlled protocol failure")

        with tempfile.TemporaryDirectory() as temp_dir:
            request = json.dumps(
                {
                    "id": "preload",
                    "type": "preload",
                    "output_dir": str(Path(temp_dir) / "outputs"),
                }
            )
            with self.assertRaisesRegex(OSError, "protocol failure"):
                run_worker(
                    io.StringIO(request + "\n"),
                    _BrokenOutput(),
                    io.StringIO(),
                    engine_factory=_PreparedFakeEngine,
                )

        self.assertEqual(_PreparedFakeEngine.instances[0].close_count, 1)


if __name__ == "__main__":
    unittest.main()
