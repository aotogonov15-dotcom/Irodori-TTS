from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from irodori_tts.inference_runtime import PreparedReferenceConditioning
from irodori_tts.local_worker import LocalWorker, run_worker
from irodori_tts.prepared_voice_cache import (
    CacheClosedError,
    PreparedVoiceCache,
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
        self.preprocessing_calls: list[dict] = []
        self.generate_calls: list[dict] = []
        self.no_ref_calls = 0
        self.load_count = 0
        self.close_count = 0
        type(self).instances.append(self)

    def load(self) -> None:
        self.load_count += 1

    def prepare_reference_conditioning(self, snapshot, **preprocessing):
        self.prepare_calls.append(snapshot)
        self.preprocessing_calls.append(preprocessing)
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

    def _create_hardlink(self, alias: Path, target: Path) -> None:
        try:
            alias.hardlink_to(target)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"hardlink creation is unavailable: {error}")
        self.assertTrue(alias.samefile(target))

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

            with patch(
                "irodori_tts.local_worker.snapshot_reference_file",
                side_effect=AssertionError("resident handle reread its fallback reference"),
            ) as snapshot:
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
        snapshot.assert_not_called()
        self.assertEqual(len(_PreparedFakeEngine.instances[0].prepare_calls), 1)

    def test_resident_handle_rejects_direct_and_normalized_reference_output_collisions(
        self,
    ) -> None:
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
            engine = _PreparedFakeEngine.instances[0]
            engine.generate_calls.clear()

            with patch(
                "irodori_tts.local_worker.snapshot_reference_file",
                side_effect=AssertionError("collision validation read the reference"),
            ) as snapshot:
                direct = harness.request(
                    self._generate(
                        reference,
                        runtime_generation=generation,
                        prepared_voice_id=prepared["prepared_voice_id"],
                        reference=self._reference(reference),
                        reference_revision=prepared["reference_revision"],
                    )
                )
                normalized = harness.request(
                    self._generate(
                        base / "unused" / ".." / reference.name,
                        runtime_generation=generation,
                        prepared_voice_id=prepared["prepared_voice_id"],
                        reference=self._reference(reference),
                        reference_revision=prepared["reference_revision"],
                    )
                )

        for response in (direct, normalized):
            self.assertEqual(response["error"]["code"], "invalid_request")
            self.assertEqual(response["error"]["reason"], "reference_is_output")
        snapshot.assert_not_called()
        self.assertEqual(engine.generate_calls, [])

    def test_stale_fallback_and_reference_only_collisions_are_rejected_before_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"fallback")
            revision = snapshot_reference_file(reference, ReferencePreprocessing()).revision
            harness = _Harness()
            generation = harness.preload(base / "outputs")["runtime_generation"]
            engine = _PreparedFakeEngine.instances[0]

            with patch(
                "irodori_tts.local_worker.snapshot_reference_file",
                side_effect=AssertionError("collision validation read the reference"),
            ) as snapshot:
                stale = harness.request(
                    self._generate(
                        reference,
                        runtime_generation=generation,
                        prepared_voice_id="f" * 32,
                        reference=self._reference(reference),
                        reference_revision=revision,
                    )
                )
                reference_only = harness.request(
                    self._generate(reference, reference=self._reference(reference))
                )

        for response in (stale, reference_only):
            self.assertEqual(response["error"]["code"], "invalid_request")
            self.assertEqual(response["error"]["reason"], "reference_is_output")
        snapshot.assert_not_called()
        self.assertEqual(engine.prepare_calls, [])
        self.assertEqual(engine.generate_calls, [])

    def test_resident_handle_rejects_hardlink_fallback_before_reference_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            output = base / "hardlink.wav"
            reference.write_bytes(b"resident-hardlink")
            self._create_hardlink(output, reference)
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
            engine = _PreparedFakeEngine.instances[0]
            engine.generate_calls.clear()

            with patch(
                "irodori_tts.local_worker.snapshot_reference_file",
                side_effect=AssertionError("hardlink validation read the reference"),
            ) as snapshot:
                response = harness.request(
                    self._generate(
                        output,
                        runtime_generation=generation,
                        prepared_voice_id=prepared["prepared_voice_id"],
                        reference=self._reference(reference),
                        reference_revision=prepared["reference_revision"],
                    )
                )

        self.assertEqual(response["error"]["code"], "invalid_request")
        self.assertEqual(response["error"]["reason"], "reference_is_output")
        snapshot.assert_not_called()
        self.assertEqual(len(engine.prepare_calls), 1)
        self.assertEqual(engine.generate_calls, [])

    def test_stale_and_reference_only_hardlink_collisions_reject_before_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            output = base / "hardlink.wav"
            reference.write_bytes(b"fallback-hardlink")
            self._create_hardlink(output, reference)
            revision = snapshot_reference_file(reference, ReferencePreprocessing()).revision
            harness = _Harness()
            generation = harness.preload(base / "outputs")["runtime_generation"]
            engine = _PreparedFakeEngine.instances[0]

            with patch(
                "irodori_tts.local_worker.snapshot_reference_file",
                side_effect=AssertionError("hardlink validation read the reference"),
            ) as snapshot:
                stale = harness.request(
                    self._generate(
                        output,
                        runtime_generation=generation,
                        prepared_voice_id="f" * 32,
                        reference=self._reference(reference),
                        reference_revision=revision,
                    )
                )
                reference_only = harness.request(
                    self._generate(output, reference=self._reference(reference))
                )

        for response in (stale, reference_only):
            self.assertEqual(response["error"]["code"], "invalid_request")
            self.assertEqual(response["error"]["reason"], "reference_is_output")
        snapshot.assert_not_called()
        self.assertEqual(engine.prepare_calls, [])
        self.assertEqual(engine.generate_calls, [])

    def test_distinct_existing_and_new_output_paths_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            existing_output = base / "existing.wav"
            new_output = base / "new.wav"
            reference.write_bytes(b"reference")
            existing_output.write_bytes(b"different")
            harness = _Harness()

            existing = harness.request(
                self._generate(existing_output, reference=self._reference(reference))
            )
            new = harness.request(self._generate(new_output, reference=self._reference(reference)))

        self.assertTrue(existing["ok"])
        self.assertTrue(new["ok"])
        self.assertEqual(len(_PreparedFakeEngine.instances[0].generate_calls), 2)

    def test_preprocessing_conflicts_use_one_merged_effective_context(self) -> None:
        accepted = (
            ("both omit", None, {}, (-16.0, False, 30.0)),
            (
                "same normalize",
                {"ref_normalize_db": -12},
                {"ref_normalize_db": -12.0},
                (-12.0, False, 30.0),
            ),
            (
                "null normalize both",
                {"ref_normalize_db": None},
                {"ref_normalize_db": None},
                (None, True, 30.0),
            ),
            (
                "null plus settings ensure",
                {"ref_normalize_db": None},
                {"ref_ensure_max": True},
                (None, True, 30.0),
            ),
            (
                "null and same ensure",
                {"ref_normalize_db": None, "ref_ensure_max": True},
                {"ref_ensure_max": True},
                (None, True, 30.0),
            ),
            (
                "new reference only",
                {"ref_normalize_db": None, "ref_ensure_max": False},
                {},
                (None, False, 30.0),
            ),
            (
                "same max seconds",
                {"max_ref_seconds": 12},
                {"max_ref_seconds": 12.0},
                (-16.0, False, 12.0),
            ),
        )
        rejected = (
            ("different normalize", {"ref_normalize_db": -12}, {"ref_normalize_db": -18}),
            (
                "null false versus true",
                {"ref_normalize_db": None, "ref_ensure_max": False},
                {"ref_ensure_max": True},
            ),
            (
                "enabled normalize ensure disagreement",
                {"ref_normalize_db": -16, "ref_ensure_max": False},
                {"ref_ensure_max": True},
            ),
            ("different max seconds", {"max_ref_seconds": 12}, {"max_ref_seconds": 15}),
        )

        for name, reference_preprocessing, settings, expected in accepted:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                _PreparedFakeEngine.instances = []
                base = Path(temp_dir)
                reference = base / "voice.wav"
                reference.write_bytes(name.encode())
                harness = _Harness()
                payload = self._generate(
                    base / "reply.wav",
                    reference=self._reference(reference, reference_preprocessing),
                    settings=settings,
                )

                response = harness.request(payload)

                self.assertTrue(response["ok"])
                preprocessing = _PreparedFakeEngine.instances[0].preprocessing_calls[0]
                self.assertEqual(
                    (
                        preprocessing["ref_normalize_db"],
                        preprocessing["ref_ensure_max"],
                        preprocessing["max_ref_seconds"],
                    ),
                    expected,
                )

        for name, reference_preprocessing, settings in rejected:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                _PreparedFakeEngine.instances = []
                base = Path(temp_dir)
                reference = base / "voice.wav"
                reference.write_bytes(name.encode())
                harness = _Harness()

                response = harness.request(
                    self._generate(
                        base / "reply.wav",
                        reference=self._reference(reference, reference_preprocessing),
                        settings=settings,
                    )
                )

                self.assertEqual(response["error"]["code"], "invalid_request")
                self.assertEqual(
                    response["error"]["reason"],
                    "conflicting_reference_preprocessing",
                )
                self.assertEqual(_PreparedFakeEngine.instances, [])

    def test_legacy_settings_keep_explicit_null_preprocessing_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"legacy-null")
            harness = _Harness()

            response = harness.request(
                self._generate(
                    base / "reply.wav",
                    reference_audio=str(reference),
                    settings={"ref_normalize_db": None, "ref_ensure_max": True},
                )
            )

        self.assertTrue(response["ok"])
        preprocessing = _PreparedFakeEngine.instances[0].preprocessing_calls[0]
        self.assertIsNone(preprocessing["ref_normalize_db"])
        self.assertIs(preprocessing["ref_ensure_max"], True)

    def test_protocol_paths_expand_home_for_legacy_b2_output_and_preload_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir) / "home"
            home.mkdir()
            legacy_reference = home / "legacy.wav"
            b2_reference = home / "b2.wav"
            legacy_reference.write_bytes(b"legacy")
            b2_reference.write_bytes(b"b2")
            with patch.dict(
                os.environ,
                {"HOME": str(home), "USERPROFILE": str(home)},
            ):
                legacy = _Harness()
                legacy_response = legacy.request(
                    self._generate(
                        Path("~/legacy-output.wav"),
                        reference_audio="~/legacy.wav",
                    )
                )
                b2 = _Harness()
                with patch(
                    "irodori_tts.local_worker.snapshot_reference_file",
                    wraps=snapshot_reference_file,
                ) as snapshot:
                    b2_response = b2.request(
                        self._generate(
                            Path("~/b2-output.wav"),
                            reference=self._reference(Path("~/b2.wav")),
                        )
                    )
                preload = _Harness()
                preload_response = preload.preload(Path("~/outputs"))

        self.assertTrue(legacy_response["ok"])
        self.assertEqual(
            _PreparedFakeEngine.instances[0].reference_audio, legacy_reference.resolve()
        )
        self.assertEqual(
            _PreparedFakeEngine.instances[0].generate_calls[0]["output_path"],
            (home / "legacy-output.wav").resolve(),
        )
        self.assertTrue(b2_response["ok"])
        self.assertEqual(snapshot.call_args.args[0], b2_reference.resolve())
        self.assertEqual(
            _PreparedFakeEngine.instances[1].generate_calls[0]["output_path"],
            (home / "b2-output.wav").resolve(),
        )
        self.assertTrue(preload_response["ok"])
        self.assertEqual(_PreparedFakeEngine.instances[2].output_dir, (home / "outputs").resolve())

    def test_absolute_and_relative_protocol_paths_keep_resolve_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"audio")
            absolute = _Harness()
            absolute_response = absolute.request(
                self._generate(base / "absolute.wav", reference_audio=str(reference))
            )
            relative_reference = os.path.relpath(reference, Path.cwd())
            relative_output = os.path.relpath(base / "relative.wav", Path.cwd())
            relative = _Harness()
            relative_response = relative.request(
                self._generate(Path(relative_output), reference_audio=relative_reference)
            )

        self.assertTrue(absolute_response["ok"])
        self.assertEqual(_PreparedFakeEngine.instances[0].reference_audio, reference.resolve())
        self.assertEqual(
            _PreparedFakeEngine.instances[0].generate_calls[0]["output_path"],
            (base / "absolute.wav").resolve(),
        )
        self.assertTrue(relative_response["ok"])
        self.assertEqual(_PreparedFakeEngine.instances[1].reference_audio, reference.resolve())
        self.assertEqual(
            _PreparedFakeEngine.instances[1].generate_calls[0]["output_path"],
            (base / "relative.wav").resolve(),
        )

    def test_home_expanded_collision_is_rejected_consistently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir) / "home"
            home.mkdir()
            reference = home / "voice.wav"
            reference.write_bytes(b"audio")
            harness = _Harness()
            with patch.dict(
                os.environ,
                {"HOME": str(home), "USERPROFILE": str(home)},
            ):
                response = harness.request(
                    self._generate(reference, reference=self._reference(Path("~/voice.wav")))
                )

        self.assertEqual(response["error"]["code"], "invalid_request")
        self.assertEqual(response["error"]["reason"], "reference_is_output")
        self.assertEqual(_PreparedFakeEngine.instances, [])

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
            cache = harness.worker.cache
            engine = _PreparedFakeEngine.instances[0]
            events = []
            original_cache_close = cache.close
            original_engine_close = engine.close

            def close_cache() -> None:
                events.append("cache")
                original_cache_close()

            def close_engine() -> None:
                events.append("engine")
                original_engine_close()

            cache.close = close_cache
            engine.close = close_engine

            response = harness.request({"id": "shutdown", "type": "shutdown"})
            harness.worker.close()

        stderr = harness.error.getvalue()
        self.assertTrue(response["ok"])
        self.assertEqual(events, ["cache", "engine"])
        self.assertLess(stderr.index("event=cache_close"), stderr.index("event=engine_close"))
        self.assertEqual(_PreparedFakeEngine.instances[0].close_count, 1)

    def test_broken_stderr_does_not_skip_cache_or_engine_cleanup(self) -> None:
        class _BrokenError(io.StringIO):
            def write(self, value):
                raise OSError("controlled diagnostic failure")

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reference = base / "voice.wav"
            reference.write_bytes(b"voice")
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
            cache = harness.worker.cache
            engine = _PreparedFakeEngine.instances[0]
            harness.worker.error_stream = _BrokenError()

            harness.worker.close()
            harness.worker.close()

        self.assertEqual(engine.close_count, 1)
        self.assertEqual(cache.entry_count, 0)
        with self.assertRaises(CacheClosedError):
            cache.acquire_handle(prepared["prepared_voice_id"])

    def test_cache_close_failure_still_attempts_engine_close_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            harness = _Harness()
            harness.preload(Path(temp_dir) / "outputs")
            cache = harness.worker.cache
            engine = _PreparedFakeEngine.instances[0]
            cache.close = Mock(side_effect=RuntimeError("cache cleanup failed"))

            with self.assertRaisesRegex(RuntimeError, "cache cleanup failed"):
                harness.worker.close()
            harness.worker.close()

        cache.close.assert_called_once_with()
        self.assertEqual(engine.close_count, 1)
        self.assertIsNone(harness.worker.cache)
        self.assertIsNone(harness.worker.engine)

    def test_engine_close_failure_keeps_terminal_bookkeeping_coherent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            harness = _Harness()
            harness.preload(Path(temp_dir) / "outputs")
            cache = harness.worker.cache
            engine = _PreparedFakeEngine.instances[0]
            engine.close = Mock(side_effect=RuntimeError("engine cleanup failed"))

            with self.assertRaisesRegex(RuntimeError, "engine cleanup failed"):
                harness.worker.close()
            harness.worker.close()

        engine.close.assert_called_once_with()
        self.assertEqual(cache.entry_count, 0)
        self.assertIsNone(harness.worker.cache)
        self.assertIsNone(harness.worker.engine)

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

        closed_caches = []
        original_cache_close = PreparedVoiceCache.close

        def tracked_cache_close(cache) -> None:
            original_cache_close(cache)
            closed_caches.append(cache)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(
                PreparedVoiceCache,
                "close",
                tracked_cache_close,
            ),
        ):
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
        self.assertEqual(len(closed_caches), 1)
        self.assertEqual(closed_caches[0].entry_count, 0)

    def test_eof_finally_closes_cache_and_engine(self) -> None:
        closed_caches = []
        original_cache_close = PreparedVoiceCache.close

        def tracked_cache_close(cache) -> None:
            original_cache_close(cache)
            closed_caches.append(cache)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(
                PreparedVoiceCache,
                "close",
                tracked_cache_close,
            ),
        ):
            request = json.dumps(
                {
                    "id": "preload",
                    "type": "preload",
                    "output_dir": str(Path(temp_dir) / "outputs"),
                }
            )
            run_worker(
                io.StringIO(request + "\n"),
                io.StringIO(),
                io.StringIO(),
                engine_factory=_PreparedFakeEngine,
            )

        self.assertEqual(_PreparedFakeEngine.instances[0].close_count, 1)
        self.assertEqual(len(closed_caches), 1)
        self.assertEqual(closed_caches[0].entry_count, 0)


if __name__ == "__main__":
    unittest.main()
