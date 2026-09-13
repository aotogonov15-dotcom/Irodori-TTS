from __future__ import annotations

import contextlib
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from irodori_tts.voice_engine import (
    VoiceEngine,
    VoiceGenerationResult,
    VoiceGenerationSettings,
)


EngineFactory = Callable[[Path | None, Path], Any]

_SETTINGS_TYPES: dict[str, type | tuple[type, ...]] = {
    "num_steps": int,
    "t_schedule_mode": str,
    "sway_coeff": (int, float),
    "cfg_guidance_mode": str,
    "cfg_scale_text": (int, float),
    "cfg_scale_speaker": (int, float),
    "duration_scale": (int, float),
    "seed": int,
    "num_candidates": int,
    "decode_mode": str,
    "ref_normalize_db": (int, float),
    "ref_ensure_max": bool,
    "max_ref_seconds": (int, float),
}
_ALLOWED_SETTINGS = set(_SETTINGS_TYPES)


class WorkerError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class LocalWorker:
    def __init__(
        self,
        *,
        output_stream: TextIO,
        error_stream: TextIO,
        engine_factory: EngineFactory,
    ) -> None:
        self.output_stream = output_stream
        self.error_stream = error_stream
        self.engine_factory = engine_factory
        self.engine: Any | None = None

    def process_line(self, line: str) -> bool:
        request_id: Any = None
        try:
            request = _parse_json_line(line)
            request_id = _extract_response_id(request)
            response, should_continue = self._handle_request(request)
        except WorkerError as error:
            response = _error_response(request_id, error.code, error.message)
            should_continue = True
        except Exception as error:  # pragma: no cover - defensive guard
            _print_traceback(self.error_stream)
            response = _error_response(
                request_id,
                "internal_error",
                _safe_error_message(error),
            )
            should_continue = True

        self._write_response(response)
        return should_continue

    def _handle_request(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        request_id = _validate_request_id(request.get("id"))
        request_type = request.get("type")

        if request_type == "shutdown":
            return {"id": request_id, "ok": True}, False
        if request_type == "preload":
            return self._handle_preload(request, request_id), True
        if request_type != "generate":
            raise WorkerError("unknown_request", "未対応のrequest typeです。")

        return self._handle_generate(request, request_id), True

    def _handle_preload(self, request: dict[str, Any], request_id: str) -> dict[str, Any]:
        reference_audio = _validate_optional_reference_audio(request.get("reference_audio"))
        output_dir = _validate_output_dir(request.get("output_dir"))
        already_loaded = self.engine is not None

        self._log(request_id, "preload_start")
        self._ensure_engine(reference_audio, output_dir)
        self._log(request_id, "preload_ok")
        return {
            "id": request_id,
            "ok": True,
            "ready": True,
            "already_loaded": already_loaded,
        }

    def _handle_generate(self, request: dict[str, Any], request_id: str) -> dict[str, Any]:
        text = _validate_text(request.get("text"))
        reference_audio = _validate_reference_audio(request.get("reference_audio"))
        output_path = _validate_output_path(request.get("output_path"))
        settings_payload = request["settings"] if "settings" in request else {}
        settings = _validate_settings(settings_payload)

        self._log(request_id, "generate_start")
        try:
            engine = self._ensure_engine(reference_audio, output_path.parent)
            result = _redirect_stdout_to_stderr(
                self.error_stream,
                engine.generate,
                text,
                reference_audio=reference_audio,
                settings=settings,
                output_path=output_path,
            )
        except WorkerError:
            raise
        except FileNotFoundError as error:
            raise WorkerError("missing_reference", "参照音声ファイルが見つかりません。") from error
        except OSError as error:
            _print_traceback(self.error_stream)
            raise WorkerError("output_save_failed", "音声ファイルの保存に失敗しました。") from error
        except RuntimeError as error:
            _print_traceback(self.error_stream)
            code = "cuda_failed" if _looks_like_cuda_error(error) else "generation_failed"
            raise WorkerError(code, _generation_error_message(code)) from error
        except Exception as error:
            _print_traceback(self.error_stream)
            raise WorkerError("generation_failed", "音声生成に失敗しました。") from error

        if not isinstance(result, VoiceGenerationResult):
            output = Path(result.output_path)
            used_seed = int(result.used_seed)
            generation_seconds = float(result.generation_seconds)
        else:
            output = result.output_path
            used_seed = result.used_seed
            generation_seconds = result.generation_seconds

        self._log(request_id, "generate_ok")
        return {
            "id": request_id,
            "ok": True,
            "output_path": str(Path(output).resolve()),
            "used_seed": used_seed,
            "generation_seconds": generation_seconds,
        }

    def _ensure_engine(self, reference_audio: Path | None, output_dir: Path) -> Any:
        if self.engine is not None:
            return self.engine

        try:
            engine = _redirect_stdout_to_stderr(
                self.error_stream,
                self.engine_factory,
                reference_audio,
                output_dir,
            )
            _redirect_stdout_to_stderr(self.error_stream, engine.load)
        except RuntimeError as error:
            _print_traceback(self.error_stream)
            code = "cuda_failed" if _looks_like_cuda_error(error) else "model_load_failed"
            raise WorkerError(code, _model_load_error_message(code)) from error
        except Exception as error:
            _print_traceback(self.error_stream)
            raise WorkerError("model_load_failed", "音声生成モデルの読み込みに失敗しました。") from error

        self.engine = engine
        return engine

    def _write_response(self, response: dict[str, Any]) -> None:
        self.output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
        self.output_stream.flush()

    def _log(self, request_id: str, event: str) -> None:
        print(f"[local_worker] id={request_id} event={event}", file=self.error_stream)


def run_worker(
    input_stream: TextIO,
    output_stream: TextIO,
    error_stream: TextIO,
    *,
    engine_factory: EngineFactory | None = None,
) -> None:
    worker = LocalWorker(
        output_stream=output_stream,
        error_stream=error_stream,
        engine_factory=engine_factory or VoiceEngine,
    )

    for line in input_stream:
        if line.strip() == "":
            continue
        should_continue = worker.process_line(line)
        if not should_continue:
            break


def main() -> None:
    _configure_standard_streams(sys.stdin, sys.stdout, sys.stderr)
    run_worker(sys.stdin, sys.stdout, sys.stderr)


def _configure_standard_streams(
    input_stream: TextIO,
    output_stream: TextIO,
    error_stream: TextIO,
) -> None:
    input_stream.reconfigure(encoding="utf-8", errors="strict")
    output_stream.reconfigure(encoding="utf-8", errors="strict")
    error_stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def _parse_json_line(line: str) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise WorkerError("invalid_json", "JSONの形式が正しくありません。") from error

    if not isinstance(payload, dict):
        raise WorkerError("invalid_request", "requestはJSON objectにしてください。")
    return payload


def _extract_response_id(request: Any) -> Any:
    if isinstance(request, dict) and isinstance(request.get("id"), str):
        return request["id"]
    return None


def _validate_request_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("invalid_request", "idは空でない文字列にしてください。")
    return value


def _validate_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("empty_text", "textは空でない文字列にしてください。")
    return value.strip()


def _validate_reference_audio(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("missing_reference", "reference_audioを指定してください。")

    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_file():
        raise WorkerError("missing_reference", "参照音声ファイルが見つかりません。")
    return path


def _validate_optional_reference_audio(value: Any) -> Path | None:
    if value is None:
        return None
    return _validate_reference_audio(value)


def _validate_output_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("invalid_request", "output_pathを指定してください。")

    path = Path(value).expanduser().resolve(strict=False)
    if path.exists() and path.is_dir():
        raise WorkerError("invalid_request", "output_pathはfile pathを指定してください。")
    if not path.name:
        raise WorkerError("invalid_request", "output_pathはfile pathを指定してください。")
    return path


def _validate_output_dir(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("invalid_request", "output_dirを指定してください。")

    path = Path(value).expanduser().resolve(strict=False)
    if path.exists() and not path.is_dir():
        raise WorkerError("invalid_request", "output_dirはdirectory pathを指定してください。")
    return path


def _validate_settings(value: Any) -> VoiceGenerationSettings:
    if not isinstance(value, dict):
        raise WorkerError("invalid_settings", "settingsはJSON objectにしてください。")

    unknown = set(value) - _ALLOWED_SETTINGS
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise WorkerError("invalid_settings", f"未対応のsettingsです: {keys}")

    coerced: dict[str, Any] = {}
    for key, raw_value in value.items():
        expected = _SETTINGS_TYPES[key]
        if expected is bool:
            if not isinstance(raw_value, bool):
                raise WorkerError("invalid_settings", f"{key}はbooleanにしてください。")
            coerced[key] = raw_value
        elif expected is int:
            if not isinstance(raw_value, int) or isinstance(raw_value, bool):
                raise WorkerError("invalid_settings", f"{key}はintegerにしてください。")
            coerced[key] = raw_value
        elif expected is str:
            if not isinstance(raw_value, str):
                raise WorkerError("invalid_settings", f"{key}はstringにしてください。")
            coerced[key] = raw_value
        else:
            if not isinstance(raw_value, expected) or isinstance(raw_value, bool):
                raise WorkerError("invalid_settings", f"{key}はnumberにしてください。")
            coerced[key] = float(raw_value)

    return VoiceGenerationSettings(**coerced)


def _redirect_stdout_to_stderr(
    error_stream: TextIO,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    with contextlib.redirect_stdout(error_stream):
        return func(*args, **kwargs)


def _error_response(request_id: Any, code: str, message: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _looks_like_cuda_error(error: BaseException) -> bool:
    error_type = type(error).__name__.lower()
    if "cuda" in error_type:
        return True
    return "cuda" in str(error).lower()


def _model_load_error_message(code: str) -> str:
    if code == "cuda_failed":
        return "CUDAの初期化または実行に失敗しました。"
    return "音声生成モデルの読み込みに失敗しました。"


def _generation_error_message(code: str) -> str:
    if code == "cuda_failed":
        return "CUDAの実行に失敗しました。"
    return "音声生成に失敗しました。"


def _safe_error_message(error: BaseException) -> str:
    if isinstance(error, WorkerError):
        return error.message
    return "worker内部で予期しないエラーが発生しました。"


def _print_traceback(error_stream: TextIO) -> None:
    traceback.print_exc(file=error_stream)


if __name__ == "__main__":
    main()
