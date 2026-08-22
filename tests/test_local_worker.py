from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from irodori_tts.local_worker import run_worker
from voice_engine import VoiceGenerationResult, VoiceGenerationSettings


class FakeVoiceEngine:
    instances: list[FakeVoiceEngine] = []
    fail_load = False

    def __init__(self, reference_audio: Path, output_dir: Path) -> None:
        self.reference_audio = reference_audio
        self.output_dir = output_dir
        self.load_count = 0
        self.generate_calls = []
        self.fail_next_generate = False
        FakeVoiceEngine.instances.append(self)
        print("factory stdout log")

    def load(self) -> None:
        self.load_count += 1
        print("load stdout log")
        if self.fail_load:
            raise RuntimeError("load failed")

    def generate(
        self,
        text: str,
        reference_audio=None,
        settings=None,
        output_path=None,
    ) -> VoiceGenerationResult:
        print("generate stdout log")
        self.generate_calls.append(
            {
                "text": text,
                "reference_audio": reference_audio,
                "settings": settings,
                "output_path": output_path,
            }
        )
        if self.fail_next_generate or text == "fail":
            self.fail_next_generate = False
            raise RuntimeError("generation failed")
        return VoiceGenerationResult(
            output_path=Path(output_path),
            used_seed=settings.seed,
            generation_seconds=0.25,
        )


class LocalWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeVoiceEngine.instances = []
        FakeVoiceEngine.fail_load = False

    def test_valid_generate_returns_json_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            reference = self._write_audio(base_dir / "reference.wav")
            output_path = base_dir / "reply.wav"

            responses, _stderr = self._run(
                [
                    self._generate_request(
                        "request-1",
                        reference_audio=reference,
                        output_path=output_path,
                    )
                ]
            )

        self.assertEqual(
            responses,
            [
                {
                    "id": "request-1",
                    "ok": True,
                    "output_path": str(output_path.resolve()),
                    "used_seed": 1234,
                    "generation_seconds": 0.25,
                }
            ],
        )

    def test_multiple_requests_reuse_loaded_engine_and_pass_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            reference1 = self._write_audio(base_dir / "reference1.wav")
            reference2 = self._write_audio(base_dir / "reference2.wav")
            output1 = base_dir / "reply1.wav"
            output2 = base_dir / "reply2.wav"

            responses, _stderr = self._run(
                [
                    self._generate_request(
                        "request-1",
                        reference_audio=reference1,
                        output_path=output1,
                    ),
                    self._generate_request(
                        "request-2",
                        text="こんばんは",
                        reference_audio=reference2,
                        output_path=output2,
                    ),
                ]
            )

        engine = FakeVoiceEngine.instances[0]
        self.assertEqual(len(FakeVoiceEngine.instances), 1)
        self.assertEqual(engine.load_count, 1)
        self.assertEqual(len(engine.generate_calls), 2)
        self.assertEqual(engine.generate_calls[0]["output_path"], output1.resolve())
        self.assertEqual(engine.generate_calls[1]["output_path"], output2.resolve())
        self.assertEqual(engine.generate_calls[1]["reference_audio"], reference2.resolve())
        self.assertTrue(all(response["ok"] for response in responses))

    def test_default_settings_are_used_when_settings_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            reference = self._write_audio(base_dir / "reference.wav")
            output_path = base_dir / "reply.wav"

            self._run(
                [
                    self._generate_request(
                        "request-1",
                        reference_audio=reference,
                        output_path=output_path,
                    )
                ]
            )

        settings = FakeVoiceEngine.instances[0].generate_calls[0]["settings"]
        self.assertEqual(settings, VoiceGenerationSettings())

    def test_custom_settings_are_mapped_to_voice_generation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            reference = self._write_audio(base_dir / "reference.wav")
            output_path = base_dir / "reply.wav"

            self._run(
                [
                    self._generate_request(
                        "request-1",
                        reference_audio=reference,
                        output_path=output_path,
                        settings={
                            "num_steps": 8,
                            "t_schedule_mode": "linear",
                            "sway_coeff": 0.25,
                            "cfg_guidance_mode": "alternating",
                            "cfg_scale_text": 2.0,
                            "cfg_scale_speaker": 4.5,
                            "duration_scale": 1.25,
                            "seed": 4321,
                            "num_candidates": 2,
                            "decode_mode": "batch",
                            "ref_normalize_db": -12.0,
                            "ref_ensure_max": False,
                            "max_ref_seconds": 12.5,
                        },
                    )
                ]
            )

        settings = FakeVoiceEngine.instances[0].generate_calls[0]["settings"]
        self.assertEqual(
            settings,
            VoiceGenerationSettings(
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
            ),
        )

    def test_unknown_settings_are_rejected_after_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            reference = self._write_audio(base_dir / "reference.wav")
            output_path = base_dir / "reply.wav"
            response = self._single_response(
                self._generate_request(
                    "request-1",
                    reference_audio=reference,
                    output_path=output_path,
                    settings={"unknown": 1},
                )
            )

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "invalid_settings")
        self.assertEqual(FakeVoiceEngine.instances, [])

    def test_non_object_settings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            reference = self._write_audio(base_dir / "reference.wav")
            output_path = base_dir / "reply.wav"
            request = self._generate_request(
                "request-1",
                reference_audio=reference,
                output_path=output_path,
            )
            request["settings"] = None
            response = self._single_response(request)

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "invalid_settings")
        self.assertEqual(FakeVoiceEngine.instances, [])

    def test_invalid_json_returns_error_with_null_id(self) -> None:
        responses, _stderr = self._run_raw("{not json}\n")

        self.assertEqual(responses[0]["id"], None)
        self.assertFalse(responses[0]["ok"])
        self.assertEqual(responses[0]["error"]["code"], "invalid_json")

    def test_non_object_json_is_rejected(self) -> None:
        responses, _stderr = self._run_raw("[]\n")

        self.assertEqual(responses[0]["id"], None)
        self.assertEqual(responses[0]["error"]["code"], "invalid_request")

    def test_unknown_request_is_rejected(self) -> None:
        response = self._single_response({"id": "request-1", "type": "ping"})

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "unknown_request")

    def test_empty_text_is_rejected(self) -> None:
        response = self._single_response({"id": "request-1", "type": "generate", "text": " "})

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "empty_text")

    def test_missing_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            response = self._single_response(
                self._generate_request(
                    "request-1",
                    reference_audio=Path(temp_dir) / "missing.wav",
                    output_path=Path(temp_dir) / "reply.wav",
                )
            )

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "missing_reference")
        self.assertEqual(FakeVoiceEngine.instances, [])

    def test_generation_failure_does_not_stop_next_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            reference = self._write_audio(base_dir / "reference.wav")
            output1 = base_dir / "reply1.wav"
            output2 = base_dir / "reply2.wav"

            responses, _stderr = self._run(
                [
                    self._generate_request(
                        "request-1",
                        text="fail",
                        reference_audio=reference,
                        output_path=output1,
                    ),
                    self._generate_request(
                        "request-2",
                        text="success",
                        reference_audio=reference,
                        output_path=output2,
                    ),
                ]
            )

        self.assertEqual(responses[0]["error"]["code"], "generation_failed")
        self.assertTrue(responses[1]["ok"])
        self.assertEqual(len(FakeVoiceEngine.instances[0].generate_calls), 2)

    def test_shutdown_returns_success_and_stops(self) -> None:
        responses, _stderr = self._run(
            [
                {"id": "shutdown-1", "type": "shutdown"},
                {"id": "request-1", "type": "ping"},
            ]
        )

        self.assertEqual(responses, [{"id": "shutdown-1", "ok": True}])

    def test_eof_without_requests_exits_without_response(self) -> None:
        responses, _stderr = self._run_raw("")

        self.assertEqual(responses, [])

    def test_stdout_is_json_lines_only_and_model_logs_go_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            reference = self._write_audio(base_dir / "reference.wav")
            output_path = base_dir / "reply.wav"

            responses, stderr = self._run(
                [
                    self._generate_request(
                        "request-1",
                        reference_audio=reference,
                        output_path=output_path,
                    )
                ],
                raw_output=True,
            )

        output_lines = responses.splitlines()
        self.assertEqual(len(output_lines), 1)
        self.assertTrue(json.loads(output_lines[0])["ok"])
        self.assertNotIn("stdout log", responses)
        self.assertIn("factory stdout log", stderr)
        self.assertIn("load stdout log", stderr)
        self.assertIn("generate stdout log", stderr)

    def _single_response(self, request: dict) -> dict:
        responses, _stderr = self._run([request])
        self.assertEqual(len(responses), 1)
        return responses[0]

    def _run(self, requests: list[dict], *, raw_output: bool = False):
        raw_input = "".join(json.dumps(request, ensure_ascii=False) + "\n" for request in requests)
        return self._run_raw(raw_input, raw_output=raw_output)

    def _run_raw(self, raw_input: str, *, raw_output: bool = False):
        input_stream = io.StringIO(raw_input)
        output_stream = io.StringIO()
        error_stream = io.StringIO()

        run_worker(
            input_stream,
            output_stream,
            error_stream,
            engine_factory=FakeVoiceEngine,
        )

        output = output_stream.getvalue()
        if raw_output:
            return output, error_stream.getvalue()
        responses = [json.loads(line) for line in output.splitlines()]
        return responses, error_stream.getvalue()

    def _generate_request(
        self,
        request_id: str,
        *,
        text: str = "こんにちは",
        reference_audio: Path,
        output_path: Path,
        settings: dict | None = None,
    ) -> dict:
        request = {
            "id": request_id,
            "type": "generate",
            "text": text,
            "reference_audio": str(reference_audio),
            "output_path": str(output_path),
        }
        if settings is not None:
            request["settings"] = settings
        return request

    def _write_audio(self, path: Path) -> Path:
        path.write_bytes(b"dummy wav")
        return path


if __name__ == "__main__":
    unittest.main()
