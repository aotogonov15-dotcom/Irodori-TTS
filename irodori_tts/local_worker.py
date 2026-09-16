from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from irodori_tts.prepared_voice_cache import (
    DEFAULT_MAX_CACHE_BYTES,
    DEFAULT_MAX_CACHE_ENTRIES,
    DEFAULT_MAX_REFERENCE_BYTES,
    CacheClosedError,
    CacheInvariantError,
    PreparedVoiceAcquisition,
    PreparedVoiceCache,
    ReferencePreprocessing,
    ReferenceSnapshot,
    ReferenceSnapshotError,
    StalePreparedVoiceHandle,
    snapshot_reference_file,
)
from irodori_tts.voice_engine import (
    CHARACTER_BACKEND,
    CHARACTER_BASE_MODEL_ID,
    VoiceEngine,
    VoiceGenerationResult,
    VoiceGenerationSettings,
)

EngineFactory = Callable[[Path | None, Path], Any]

PROTOCOL_VERSION = 4
CAPABILITIES = ("no_ref", "prepared_voice_handles", "character_sessions")
_EXPLICIT_CHARACTER_SESSION_MODE = "explicit"
_LEGACY_CHARACTER_SESSION_MODE = "legacy"
_ABSENT = object()
_HANDLE_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{64}$")

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
    "ref_normalize_db": (int, float, type(None)),
    "ref_ensure_max": bool,
    "max_ref_seconds": (int, float),
}
_ALLOWED_SETTINGS = set(_SETTINGS_TYPES)
_PREPROCESSING_FIELDS = {"ref_normalize_db", "ref_ensure_max", "max_ref_seconds"}


class WorkerError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.reason = reason


@dataclass(frozen=True)
class _ReferenceRequest:
    path: Path
    preprocessing: ReferencePreprocessing
    legacy: bool = False


@dataclass(frozen=True)
class _CharacterVoiceIdentity:
    kind: str
    lora_path_identity: str | None = None
    compatibility_backend: str | None = None
    compatibility_base_model_id: str | None = None


@dataclass(frozen=True)
class _CharacterVoiceRequest:
    identity: _CharacterVoiceIdentity
    lora_resource_path: Path | None = None


@dataclass(frozen=True)
class _CharacterClaim:
    runtime_generation: str
    character_session_id: str
    voice: _CharacterVoiceIdentity


class LocalWorker:
    def __init__(
        self,
        *,
        output_stream: TextIO,
        error_stream: TextIO,
        engine_factory: EngineFactory,
        cache_max_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
        cache_max_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        max_reference_bytes: int = DEFAULT_MAX_REFERENCE_BYTES,
    ) -> None:
        self.output_stream = output_stream
        self.error_stream = error_stream
        self.engine_factory = engine_factory
        self.cache_max_entries = cache_max_entries
        self.cache_max_bytes = cache_max_bytes
        self.max_reference_bytes = max_reference_bytes
        self.engine: Any | None = None
        self.cache: PreparedVoiceCache | None = None
        self._closed = False
        self._request_lock = threading.RLock()
        self._character_session_mode: str | None = None
        self._character_claim: _CharacterClaim | None = None
        self._claim_reservation: _CharacterClaim | None = None
        self._claim_failed = False
        self._character_claim_path_base = _capture_character_claim_path_base()

    @property
    def runtime_generation(self) -> str | None:
        if self.engine is None:
            return None
        generation = getattr(self.engine, "runtime_generation", None)
        return generation if isinstance(generation, str) and generation else None

    def process_line(self, line: str) -> bool:
        with self._request_lock:
            return self._process_line(line)

    def _process_line(self, line: str) -> bool:
        request_id: Any = None
        try:
            request = _parse_json_line(line)
            request_id = _extract_response_id(request)
            response, should_continue = self._handle_request(request)
        except WorkerError as error:
            response = _error_response(
                request_id,
                error.code,
                error.message,
                stage=error.stage,
                reason=error.reason,
            )
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

    def close(self) -> None:
        with self._request_lock:
            self._close()

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cache = self.cache
        self.cache = None
        engine = self.engine
        self.engine = None
        had_session_state = self._character_session_mode is not None
        self._character_claim = None
        self._claim_reservation = None
        self._character_session_mode = None
        self._claim_failed = False
        cleanup_errors: list[BaseException] = []
        if cache is not None:
            entries = None
            estimated_bytes = None
            with contextlib.suppress(Exception):
                entries = cache.entry_count
                estimated_bytes = cache.estimated_bytes
            try:
                cache.close()
            except BaseException as error:
                cleanup_errors.append(error)
            self._log_best_effort(
                "worker",
                "cache_close",
                entries=entries,
                estimated_bytes=estimated_bytes,
            )
        if engine is not None:
            try:
                _redirect_stdout_to_stderr(self.error_stream, engine.close)
            except BaseException as error:
                cleanup_errors.append(error)
            self._log_best_effort("worker", "engine_close")
        if cache is not None or engine is not None or had_session_state:
            self._log_best_effort("worker", "close")
        if cleanup_errors:
            raise cleanup_errors[0]

    def _handle_request(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        request_id = _validate_request_id(request.get("id"))
        request_type = request.get("type")

        if request_type == "shutdown":
            try:
                self.close()
            except Exception:
                _print_traceback(self.error_stream)
                return (
                    _error_response(
                        request_id,
                        "runtime_unavailable",
                        "workerの終了処理に失敗しました。",
                        stage="shutdown",
                        reason="close_failed",
                    ),
                    False,
                )
            return {"id": request_id, "ok": True}, False
        if self._closed:
            raise WorkerError(
                "runtime_unavailable",
                "workerは終了処理済みです。",
                stage="request",
                reason="worker_closed",
            )
        if request_type == "preload":
            return self._handle_preload(request, request_id), True
        if request_type == "claim_character":
            return self._handle_claim_character(request, request_id), True
        if request_type == "prepare_voice":
            return self._handle_prepare_voice(request, request_id), True
        if request_type != "generate":
            raise WorkerError("unknown_request", "未対応のrequest typeです。")

        return self._handle_generate(request, request_id), True

    def _handle_preload(self, request: dict[str, Any], request_id: str) -> dict[str, Any]:
        requested_mode = _validate_character_session_mode(
            request.get("character_session_mode", _ABSENT)
        )
        mode_was_unset = self._character_session_mode is None
        self._fix_character_session_mode(requested_mode)
        if self._claim_failed:
            raise WorkerError(
                "runtime_unavailable",
                "character claim失敗後のRuntimeは再利用できません。",
                stage="preload",
                reason="previous_claim_failed",
            )
        reference_audio = _validate_optional_reference_audio(request.get("reference_audio"))
        output_dir = _validate_output_dir(request.get("output_dir"))
        already_loaded = self.engine is not None

        if mode_was_unset and requested_mode == _EXPLICIT_CHARACTER_SESSION_MODE:
            self._log(request_id, "explicit_mode_enabled")
        self._log(request_id, "preload_start", character_session_mode=requested_mode)
        self._ensure_engine(reference_audio, output_dir)
        generation = self._require_current_generation()
        character_state = self._current_character_state()
        self._log(
            request_id,
            "preload_ok",
            runtime_generation=generation,
            character_session_mode=requested_mode,
            character_state=character_state,
        )
        return {
            "id": request_id,
            "ok": True,
            "ready": True,
            "already_loaded": already_loaded,
            "protocol_version": PROTOCOL_VERSION,
            "capabilities": list(CAPABILITIES),
            "runtime_generation": generation,
            "character_session_mode": requested_mode,
            "character_state": character_state,
        }

    def _handle_claim_character(
        self,
        request: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        unsupported = set(request) - {
            "id",
            "type",
            "runtime_generation",
            "character_session_id",
            "character_voice",
        }
        if unsupported:
            raise WorkerError(
                "invalid_request",
                "claim_characterに未対応fieldを指定できません。",
                stage="validation",
                reason="unsupported_claim_field",
            )
        if self._character_session_mode != _EXPLICIT_CHARACTER_SESSION_MODE:
            raise WorkerError(
                "runtime_unavailable",
                "explicit character session preloadを先に実行してください。",
                stage="claim",
                reason="explicit_mode_not_enabled",
            )
        engine = self._require_engine()
        generation = self._validate_runtime_generation(request, required=True)
        character_session_id = _validate_character_session_id(request.get("character_session_id"))
        if self._claim_failed:
            raise WorkerError(
                "runtime_unavailable",
                "character claim失敗後のRuntimeは再利用できません。",
                stage="claim",
                reason="previous_claim_failed",
            )
        voice_request = _normalize_character_voice(
            request.get("character_voice"),
            path_base=self._character_claim_path_base,
        )
        voice = voice_request.identity
        requested_claim = _CharacterClaim(generation, character_session_id, voice)

        existing_claim = self._character_claim
        if existing_claim is not None:
            if existing_claim == requested_claim:
                self._log(
                    request_id,
                    "character_claim_idempotent",
                    runtime_generation=generation,
                    voice_kind=voice.kind,
                )
                return self._claim_response(request_id, existing_claim, already_claimed=True)
            self._log(
                request_id,
                "character_claim_conflict",
                runtime_generation=generation,
                voice_kind=voice.kind,
            )
            raise WorkerError(
                "character_claim_conflict",
                "Runtimeは別のimmutable character claimで既にlockされています。",
                stage="claim",
                reason=(
                    "different_character_session"
                    if existing_claim.character_session_id != character_session_id
                    else "different_character_voice"
                ),
            )

        self._claim_reservation = requested_claim
        self._log(
            request_id,
            "character_claim_start",
            runtime_generation=generation,
            voice_kind=voice.kind,
        )
        try:
            _validate_character_voice_resources(voice_request)
            _redirect_stdout_to_stderr(
                self.error_stream,
                engine.claim_character,
                voice_request.lora_resource_path,
            )
        except WorkerError as error:
            self._claim_failed = True
            self._log(
                request_id,
                "character_claim_failure",
                runtime_generation=generation,
                voice_kind=voice.kind,
                reason=error.reason,
            )
            raise
        except Exception as error:
            self._claim_failed = True
            self._log(
                request_id,
                "character_claim_failure",
                runtime_generation=generation,
                voice_kind=voice.kind,
                error_type=type(error).__name__,
            )
            raise WorkerError(
                "invalid_character_voice",
                "character voiceをRuntimeに適用できませんでした。",
                stage="claim",
                reason="character_finalize_failed",
            ) from error
        finally:
            self._claim_reservation = None

        self._character_claim = requested_claim
        self._log(
            request_id,
            "character_claim_success",
            runtime_generation=generation,
            voice_kind=voice.kind,
        )
        return self._claim_response(request_id, requested_claim, already_claimed=False)

    @staticmethod
    def _claim_response(
        request_id: str,
        claim: _CharacterClaim,
        *,
        already_claimed: bool,
    ) -> dict[str, Any]:
        return {
            "id": request_id,
            "ok": True,
            "runtime_generation": claim.runtime_generation,
            "character_session_id": claim.character_session_id,
            "character_state": "character_locked",
            "voice_kind": claim.voice.kind,
            "already_claimed": already_claimed,
        }

    def _handle_prepare_voice(
        self,
        request: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        if "no_ref" in request:
            raise WorkerError(
                "invalid_request",
                "prepare_voiceではno_refを使用できません。",
                stage="validation",
                reason="no_ref_not_preparable",
            )
        unsupported = set(request) - {
            "id",
            "type",
            "runtime_generation",
            "character_session_id",
            "reference",
        }
        if unsupported:
            raise WorkerError(
                "invalid_request",
                "prepare_voiceに未対応fieldを指定できません。",
                stage="validation",
                reason="unsupported_prepare_field",
            )
        engine = self._require_engine()
        generation = self._validate_character_access(request, generation_required=True)
        reference = _validate_reference_request(request.get("reference"))
        snapshot = self._snapshot(reference)
        acquisition, duration = self._prepare_snapshot(engine, snapshot)
        if not acquisition.cached:
            self._log_cache_result(request_id, acquisition, duration, snapshot.revision)
            raise WorkerError(
                "prepare_failed",
                "Prepared voiceがcache容量を超えています。",
                stage="prepare",
                reason="capacity_exceeded",
            )
        self._log_cache_result(request_id, acquisition, duration, snapshot.revision)
        return {
            "id": request_id,
            "ok": True,
            "prepared_voice_id": acquisition.entry.prepared_voice_id,
            "runtime_generation": generation,
            "reference_revision": snapshot.revision,
            "cache_hit": acquisition.cache_hit,
        }

    def _handle_generate(self, request: dict[str, Any], request_id: str) -> dict[str, Any]:
        text = _validate_text(request.get("text"))
        output_path = _validate_output_path(request.get("output_path"))
        settings_payload = request["settings"] if "settings" in request else {}
        settings = _validate_settings(settings_payload)
        no_ref = _validate_no_ref(request.get("no_ref", False))
        raw_handle = request.get("prepared_voice_id")
        has_handle = raw_handle is not None
        has_reference = request.get("reference") is not None
        has_legacy_reference = request.get("reference_audio") is not None

        if has_reference and has_legacy_reference:
            raise _contradictory("reference_and_reference_audio")
        if no_ref and (has_handle or has_reference or has_legacy_reference):
            raise _contradictory("no_ref_with_reference")
        if has_handle and has_legacy_reference:
            raise _contradictory("handle_with_legacy_reference")
        if not no_ref and not has_handle and not has_reference and not has_legacy_reference:
            raise WorkerError("missing_reference", "参照音声を指定してください。")
        if "reference_revision" in request and not (has_handle and has_reference):
            raise _contradictory("unexpected_reference_revision")
        if has_handle and has_reference and "reference_revision" not in request:
            raise _contradictory("missing_reference_revision")
        reference = None
        if has_reference:
            reference = _validate_reference_request(
                request["reference"],
                settings_payload=settings_payload,
                settings=settings,
            )
        elif has_legacy_reference:
            reference = _ReferenceRequest(
                path=_validate_reference_audio_path(request["reference_audio"]),
                preprocessing=_preprocessing_from_settings(settings),
                legacy=True,
            )

        if reference is not None:
            _reject_reference_output_collisions((reference.path,), output_path)
            if reference.legacy:
                _require_reference_audio_file(reference.path)

        fallback_revision = (
            _validate_reference_revision(request.get("reference_revision"))
            if has_handle and has_reference
            else None
        )
        handle = _validate_prepared_voice_id(raw_handle) if has_handle else None

        if self._character_session_mode is None:
            self._fix_character_session_mode(_LEGACY_CHARACTER_SESSION_MODE)
        self._log(request_id, "generate_start", runtime_generation=self.runtime_generation)
        engine = (
            self._require_engine()
            if has_handle
            else self._ensure_engine(
                None if reference is None else reference.path,
                output_path.parent,
            )
        )
        generation = self._validate_character_access(
            request,
            generation_required=has_handle,
        )

        if no_ref:
            result = self._run_generate(
                engine.generate_no_ref,
                text,
                settings=settings,
                output_path=output_path,
            )
            return self._generation_response(
                request_id,
                result,
                generation=generation,
                prepared_voice_id=None,
                reference_revision_value=None,
                cached=False,
                cache_hit=False,
            )

        acquisition: PreparedVoiceAcquisition
        revision: str
        if has_handle:
            cache = self._require_cache()
            try:
                entry = cache.acquire_handle(handle)
                acquisition = PreparedVoiceAcquisition(
                    prepared=entry.prepared,
                    entry=entry,
                    cache_hit=True,
                    estimated_bytes=entry.estimated_bytes,
                )
                revision = entry.key.reference_revision
                self._log(
                    request_id,
                    "cache_hit",
                    runtime_generation=generation,
                    entries=cache.entry_count,
                    estimated_bytes=cache.estimated_bytes,
                )
            except StalePreparedVoiceHandle:
                self._log(request_id, "stale_handle", runtime_generation=generation)
                if reference is None:
                    raise WorkerError(
                        "stale_handle",
                        "prepared_voice_idは存在しないか失効しています。",
                        stage="handle_resolution",
                        reason="not_resident",
                    ) from None
                snapshot = self._snapshot(reference)
                if snapshot.revision != fallback_revision:
                    raise WorkerError(
                        "invalid_reference",
                        "fallback参照音声が期待されたrevisionと一致しません。",
                        stage="fallback",
                        reason="revision_mismatch",
                    ) from None
                self._log(request_id, "fallback", runtime_generation=generation)
                acquisition, duration = self._prepare_snapshot(engine, snapshot)
                self._log_cache_result(request_id, acquisition, duration, snapshot.revision)
                revision = snapshot.revision
            except (CacheClosedError, CacheInvariantError) as error:
                raise WorkerError(
                    "runtime_unavailable",
                    "Prepared voice cacheを利用できません。",
                    stage="handle_resolution",
                    reason="cache_invariant",
                ) from error
        else:
            if reference is None:  # pragma: no cover - guarded above
                raise CacheInvariantError("Reference form was lost after validation.")
            snapshot = self._snapshot(reference)
            acquisition, duration = self._prepare_snapshot(engine, snapshot)
            self._log_cache_result(request_id, acquisition, duration, snapshot.revision)
            revision = snapshot.revision

        try:
            result = self._run_generate(
                engine.generate_with_prepared,
                text,
                acquisition.prepared,
                settings=settings,
                output_path=output_path,
            )
        except WorkerError as error:
            if "runtime-owner mismatch" in str(error.__cause__):
                raise WorkerError(
                    "runtime_unavailable",
                    "Prepared voiceのRuntime ownership検証に失敗しました。",
                    stage="synthesize",
                    reason="owner_mismatch",
                ) from error
            raise

        return self._generation_response(
            request_id,
            result,
            generation=generation,
            prepared_voice_id=(
                None if acquisition.entry is None else acquisition.entry.prepared_voice_id
            ),
            reference_revision_value=revision,
            cached=acquisition.cached,
            cache_hit=acquisition.cache_hit,
        )

    def _prepare_snapshot(
        self,
        engine: Any,
        snapshot: ReferenceSnapshot,
    ) -> tuple[PreparedVoiceAcquisition, float]:
        cache = self._require_cache()
        key = cache.cache_key(snapshot.revision)
        started = time.perf_counter()
        try:
            acquisition = cache.get_or_prepare(
                key,
                lambda: _redirect_stdout_to_stderr(
                    self.error_stream,
                    engine.prepare_reference_conditioning,
                    snapshot.data,
                    ref_normalize_db=snapshot.preprocessing.ref_normalize_db,
                    ref_ensure_max=snapshot.preprocessing.ref_ensure_max,
                    max_ref_seconds=snapshot.preprocessing.max_ref_seconds,
                ),
            )
        except CacheClosedError as error:
            raise WorkerError(
                "runtime_unavailable",
                "Prepared voice cacheは終了処理済みです。",
                stage="prepare",
                reason="cache_closed",
            ) from error
        except CacheInvariantError as error:
            raise WorkerError(
                "runtime_unavailable",
                "Prepared voice cacheのRuntime整合性が失われました。",
                stage="prepare",
                reason="cache_invariant",
            ) from error
        except Exception as error:
            _print_traceback(self.error_stream)
            raise WorkerError(
                "prepare_failed",
                "参照音声の準備に失敗しました。",
                stage="prepare",
                reason="runtime_prepare_failed",
            ) from error
        return acquisition, time.perf_counter() - started

    def _snapshot(self, reference: _ReferenceRequest) -> ReferenceSnapshot:
        try:
            return snapshot_reference_file(
                reference.path,
                reference.preprocessing,
                max_bytes=self.max_reference_bytes,
            )
        except ReferenceSnapshotError as error:
            code = (
                "missing_reference"
                if reference.legacy and error.reason == "not_found"
                else "invalid_reference"
            )
            raise WorkerError(
                code,
                "参照音声ファイルを読み込めません。",
                stage="snapshot",
                reason=error.reason,
            ) from error

    def _run_generate(self, generate: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return _redirect_stdout_to_stderr(
                self.error_stream,
                generate,
                *args,
                **kwargs,
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
            raise WorkerError(code, _generation_error_message(code), stage="synthesize") from error
        except Exception as error:
            _print_traceback(self.error_stream)
            raise WorkerError(
                "generation_failed",
                "音声生成に失敗しました。",
                stage="synthesize",
            ) from error

    def _generation_response(
        self,
        request_id: str,
        result: Any,
        *,
        generation: str,
        prepared_voice_id: str | None,
        reference_revision_value: str | None,
        cached: bool,
        cache_hit: bool,
    ) -> dict[str, Any]:
        if not isinstance(result, VoiceGenerationResult):
            output = Path(result.output_path)
            used_seed = int(result.used_seed)
            generation_seconds = float(result.generation_seconds)
        else:
            output = result.output_path
            used_seed = result.used_seed
            generation_seconds = result.generation_seconds

        self._log(request_id, "generate_ok", runtime_generation=generation)
        return {
            "id": request_id,
            "ok": True,
            "output_path": str(Path(output).resolve()),
            "used_seed": used_seed,
            "generation_seconds": generation_seconds,
            "prepared_voice_id": prepared_voice_id,
            "runtime_generation": generation,
            "reference_revision": reference_revision_value,
            "cached": cached,
            "cache_hit": cache_hit,
        }

    def _ensure_engine(self, reference_audio: Path | None, output_dir: Path) -> Any:
        if self.engine is not None:
            self._require_cache()
            return self.engine
        if self._closed:
            raise WorkerError(
                "runtime_unavailable",
                "workerは終了処理済みです。",
                stage="load",
                reason="worker_closed",
            )

        engine = None
        try:
            engine = _redirect_stdout_to_stderr(
                self.error_stream,
                self.engine_factory,
                reference_audio,
                output_dir,
            )
            _redirect_stdout_to_stderr(self.error_stream, engine.load)
            generation = getattr(engine, "runtime_generation", None)
            if not isinstance(generation, str) or not generation:
                raise RuntimeError("VoiceEngine did not expose a runtime_generation.")
        except RuntimeError as error:
            if engine is not None and hasattr(engine, "close"):
                with contextlib.suppress(Exception):
                    _redirect_stdout_to_stderr(self.error_stream, engine.close)
            _print_traceback(self.error_stream)
            code = "cuda_failed" if _looks_like_cuda_error(error) else "model_load_failed"
            raise WorkerError(code, _model_load_error_message(code), stage="load") from error
        except Exception as error:
            if engine is not None and hasattr(engine, "close"):
                with contextlib.suppress(Exception):
                    _redirect_stdout_to_stderr(self.error_stream, engine.close)
            _print_traceback(self.error_stream)
            raise WorkerError(
                "model_load_failed",
                "音声生成モデルの読み込みに失敗しました。",
                stage="load",
            ) from error

        self.engine = engine
        self.cache = PreparedVoiceCache(
            generation,
            max_entries=self.cache_max_entries,
            max_bytes=self.cache_max_bytes,
        )
        return engine

    def _require_engine(self) -> Any:
        if self.engine is None:
            raise WorkerError(
                "runtime_unavailable",
                "先にpreloadを実行してください。",
                stage="runtime",
                reason="not_loaded",
            )
        return self.engine

    def _require_cache(self) -> PreparedVoiceCache:
        if self.cache is None:
            raise WorkerError(
                "runtime_unavailable",
                "Prepared voice cacheを利用できません。",
                stage="runtime",
                reason="cache_unavailable",
            )
        if self.cache.runtime_generation != self._require_current_generation():
            raise WorkerError(
                "runtime_unavailable",
                "Runtime generationとcache namespaceが一致しません。",
                stage="runtime",
                reason="cache_generation_mismatch",
            )
        return self.cache

    def _require_current_generation(self) -> str:
        generation = self.runtime_generation
        if generation is None:
            raise WorkerError(
                "runtime_unavailable",
                "Runtime generationを利用できません。",
                stage="runtime",
                reason="generation_unavailable",
            )
        return generation

    def _fix_character_session_mode(self, requested_mode: str) -> None:
        current_mode = self._character_session_mode
        if current_mode is None:
            self._character_session_mode = requested_mode
            return
        if current_mode != requested_mode:
            raise WorkerError(
                "invalid_request",
                "worker lifetime中にcharacter_session_modeを変更できません。",
                stage="preload",
                reason="character_session_mode_conflict",
            )

    def _current_character_state(self) -> str:
        if self._claim_failed:
            return "failed"
        if self._character_claim is not None:
            return "character_locked"
        if self._character_session_mode == _EXPLICIT_CHARACTER_SESSION_MODE:
            return "unclaimed"
        engine = self.engine
        if engine is not None:
            state = getattr(engine, "character_state", None)
            if state in {"character_locked", "failed"}:
                return state
        return "unclaimed"

    def _validate_character_access(
        self,
        request: dict[str, Any],
        *,
        generation_required: bool,
    ) -> str:
        if self._character_session_mode != _EXPLICIT_CHARACTER_SESSION_MODE:
            return self._validate_runtime_generation(request, required=generation_required)
        if self._claim_failed:
            raise WorkerError(
                "runtime_unavailable",
                "character claim失敗後のRuntimeは利用できません。",
                stage="runtime",
                reason="previous_claim_failed",
            )
        claim = self._character_claim
        if claim is None:
            raise WorkerError(
                "character_not_locked",
                "claim_characterを先に成功させてください。",
                stage="character_session",
                reason="character_not_locked",
            )
        generation = self._validate_runtime_generation(request, required=True)
        supplied_session_id = request.get("character_session_id")
        if not isinstance(supplied_session_id, str) or not supplied_session_id.strip():
            raise WorkerError(
                "character_session_mismatch",
                "character_session_idを指定してください。",
                stage="character_session",
                reason="missing_character_session_id",
            )
        if supplied_session_id != claim.character_session_id:
            raise WorkerError(
                "character_session_mismatch",
                "character_session_idが現在のclaimと一致しません。",
                stage="character_session",
                reason="character_session_mismatch",
            )
        return generation

    def _validate_runtime_generation(
        self,
        request: dict[str, Any],
        *,
        required: bool,
    ) -> str:
        current = self._require_current_generation()
        supplied = request.get("runtime_generation")
        if supplied is None:
            if required:
                raise WorkerError(
                    "invalid_request",
                    "runtime_generationを指定してください。",
                    stage="validation",
                    reason="missing_runtime_generation",
                )
            return current
        if not isinstance(supplied, str) or not supplied:
            raise WorkerError(
                "invalid_request",
                "runtime_generationは空でない文字列にしてください。",
                stage="validation",
                reason="invalid_runtime_generation",
            )
        if supplied != current:
            raise WorkerError(
                "runtime_unavailable",
                "runtime_generationが現在のRuntimeと一致しません。",
                stage="runtime",
                reason="runtime_generation_mismatch",
            )
        return current

    def _log_cache_result(
        self,
        request_id: str,
        acquisition: PreparedVoiceAcquisition,
        duration: float,
        revision: str,
    ) -> None:
        cache = self._require_cache()
        event = "cache_hit" if acquisition.cache_hit else "cache_miss"
        self._log(
            request_id,
            event,
            runtime_generation=cache.runtime_generation,
            reference_revision=revision[:12],
            prepare_ms=round(duration * 1000.0, 1),
            entries=cache.entry_count,
            estimated_bytes=cache.estimated_bytes,
            evicted=acquisition.evicted_count,
            coalesced_waiters=acquisition.coalesced_waiters,
            cached=acquisition.cached,
        )
        if not acquisition.cached:
            self._log(
                request_id,
                "capacity_exceeded_uncached",
                prepared_bytes=acquisition.estimated_bytes,
            )
        if acquisition.evicted_count:
            self._log(request_id, "eviction", count=acquisition.evicted_count)

    def _write_response(self, response: dict[str, Any]) -> None:
        self.output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
        self.output_stream.flush()

    def _log(self, request_id: str, event: str, **details: Any) -> None:
        suffix = "".join(f" {key}={value}" for key, value in details.items())
        print(
            f"[local_worker] id={request_id} event={event}{suffix}",
            file=self.error_stream,
        )

    def _log_best_effort(self, request_id: str, event: str, **details: Any) -> None:
        with contextlib.suppress(Exception):
            self._log(request_id, event, **details)


def run_worker(
    input_stream: TextIO,
    output_stream: TextIO,
    error_stream: TextIO,
    *,
    engine_factory: EngineFactory | None = None,
    cache_max_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
    cache_max_bytes: int = DEFAULT_MAX_CACHE_BYTES,
    max_reference_bytes: int = DEFAULT_MAX_REFERENCE_BYTES,
) -> None:
    """Execute already-sent JSON requests sequentially on one worker lane."""
    worker = LocalWorker(
        output_stream=output_stream,
        error_stream=error_stream,
        engine_factory=engine_factory or VoiceEngine,
        cache_max_entries=cache_max_entries,
        cache_max_bytes=cache_max_bytes,
        max_reference_bytes=max_reference_bytes,
    )

    try:
        for line in input_stream:
            if line.strip() == "":
                continue
            should_continue = worker.process_line(line)
            if not should_continue:
                break
    finally:
        try:
            worker.close()
        except Exception:
            _print_traceback(error_stream)


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


def _validate_character_session_mode(value: Any) -> str:
    if value is _ABSENT:
        return _LEGACY_CHARACTER_SESSION_MODE
    if value == _EXPLICIT_CHARACTER_SESSION_MODE:
        return _EXPLICIT_CHARACTER_SESSION_MODE
    raise WorkerError(
        "invalid_request",
        "character_session_modeはexplicitのみ指定できます。",
        stage="validation",
        reason="invalid_character_session_mode",
    )


def _validate_character_session_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError(
            "invalid_request",
            "character_session_idは空でない文字列にしてください。",
            stage="validation",
            reason="invalid_character_session_id",
        )
    return value


def _normalize_character_voice(
    value: Any,
    *,
    path_base: str,
) -> _CharacterVoiceRequest:
    if not isinstance(value, dict):
        raise _invalid_character_voice("invalid_character_voice_object")
    kind = value.get("kind")
    if kind == "base":
        if set(value) != {"kind"}:
            raise _invalid_character_voice("unsupported_base_voice_field")
        return _CharacterVoiceRequest(_CharacterVoiceIdentity(kind="base"))
    if kind != "lora":
        raise _invalid_character_voice("invalid_voice_kind")
    if set(value) != {"kind", "path", "compatibility"}:
        raise _invalid_character_voice("invalid_lora_voice_fields")

    compatibility = value.get("compatibility")
    if not isinstance(compatibility, dict) or set(compatibility) != {
        "backend",
        "base_model_id",
    }:
        raise _invalid_character_voice("invalid_compatibility")
    backend = compatibility.get("backend")
    base_model_id = compatibility.get("base_model_id")
    if not isinstance(backend, str) or not isinstance(base_model_id, str):
        raise _invalid_character_voice("invalid_compatibility")

    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise _invalid_character_voice("missing_lora_path")
    path_identity, resource_path = _normalize_character_claim_path(
        raw_path,
        path_base=path_base,
    )
    return _CharacterVoiceRequest(
        identity=_CharacterVoiceIdentity(
            kind="lora",
            lora_path_identity=path_identity,
            compatibility_backend=backend,
            compatibility_base_model_id=base_model_id,
        ),
        lora_resource_path=resource_path,
    )


def _capture_character_claim_path_base() -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.curdir)))


def _normalize_character_claim_path(value: str, *, path_base: str) -> tuple[str, Path]:
    expanded = os.path.expanduser(value)
    absolute = os.path.join(path_base, expanded)
    normalized = os.path.normcase(os.path.normpath(absolute))
    return normalized, Path(normalized)


def _validate_character_voice_resources(voice_request: _CharacterVoiceRequest) -> None:
    voice = voice_request.identity
    if voice.kind == "base":
        return
    if voice.compatibility_backend != CHARACTER_BACKEND:
        raise _invalid_character_voice("compatibility_backend_mismatch")
    if voice.compatibility_base_model_id != CHARACTER_BASE_MODEL_ID:
        raise _invalid_character_voice("compatibility_base_model_mismatch")
    path = voice_request.lora_resource_path
    if path is None or not path.is_dir():
        raise _invalid_character_voice("lora_path_not_directory")


def _invalid_character_voice(reason: str) -> WorkerError:
    return WorkerError(
        "invalid_character_voice",
        "character_voiceの指定が不正または現在のRuntimeと非互換です。",
        stage="validation",
        reason=reason,
    )


def _validate_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("empty_text", "textは空でない文字列にしてください。")
    return value.strip()


def _validate_reference_audio(value: Any) -> Path:
    path = _validate_reference_audio_path(value)
    _require_reference_audio_file(path)
    return path


def _validate_reference_audio_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("missing_reference", "reference_audioを指定してください。")
    return _resolve_protocol_path(value)


def _require_reference_audio_file(path: Path) -> None:
    if not path.is_file():
        raise WorkerError("missing_reference", "参照音声ファイルが見つかりません。")


def _validate_optional_reference_audio(value: Any) -> Path | None:
    if value is None:
        return None
    return _validate_reference_audio(value)


def _validate_reference_request(
    value: Any,
    *,
    settings_payload: dict[str, Any] | None = None,
    settings: VoiceGenerationSettings | None = None,
) -> _ReferenceRequest:
    if not isinstance(value, dict):
        raise WorkerError(
            "invalid_reference",
            "referenceはJSON objectにしてください。",
            stage="validation",
            reason="invalid_reference_object",
        )
    unknown = set(value) - {"files", "preprocessing"}
    if unknown:
        raise WorkerError(
            "invalid_reference",
            "referenceに未対応fieldがあります。",
            stage="validation",
            reason="unknown_reference_field",
        )
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise WorkerError(
            "invalid_reference",
            "v3 reference.filesには1つのpathを指定してください。",
            stage="validation",
            reason="file_count",
        )
    raw_path = files[0]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise WorkerError(
            "invalid_reference",
            "reference file pathは空でない文字列にしてください。",
            stage="validation",
            reason="invalid_path",
        )
    path = _resolve_protocol_path(raw_path)
    preprocessing = _effective_reference_preprocessing(
        value,
        settings_payload=settings_payload,
        settings=settings,
    )
    return _ReferenceRequest(path=path, preprocessing=preprocessing)


def _validate_explicit_preprocessing(value: Any) -> dict[str, float | bool | None]:
    if not isinstance(value, dict):
        raise WorkerError(
            "invalid_reference",
            "reference.preprocessingはJSON objectにしてください。",
            stage="validation",
            reason="invalid_preprocessing",
        )
    unknown = set(value) - _PREPROCESSING_FIELDS
    if unknown:
        raise WorkerError(
            "invalid_reference",
            "reference.preprocessingに未対応fieldがあります。",
            stage="validation",
            reason="unknown_preprocessing_field",
        )
    validated: dict[str, float | bool | None] = {}
    for name, raw_value in value.items():
        if name == "ref_ensure_max":
            if not isinstance(raw_value, bool):
                raise _invalid_preprocessing_value(name)
            validated[name] = raw_value
        elif raw_value is None:
            validated[name] = None
        elif not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            raise _invalid_preprocessing_value(name)
        else:
            validated[name] = float(raw_value)
    return validated


def _effective_reference_preprocessing(
    reference: dict[str, Any],
    *,
    settings_payload: dict[str, Any] | None,
    settings: VoiceGenerationSettings | None,
) -> ReferencePreprocessing:
    """Merge raw request fields once, then canonicalize effective cache identity."""
    explicit = _validate_explicit_preprocessing(reference.get("preprocessing", {}))
    if settings is None:
        merged: dict[str, float | bool | None] = {
            "ref_normalize_db": -16.0,
            "ref_ensure_max": True,
            "max_ref_seconds": 30.0,
        }
    else:
        merged = {
            "ref_normalize_db": settings.ref_normalize_db,
            "ref_ensure_max": settings.ref_ensure_max,
            "max_ref_seconds": settings.max_ref_seconds,
        }
        for name in explicit.keys() & (settings_payload or {}).keys():
            if explicit[name] != getattr(settings, name):
                raise _contradictory("conflicting_reference_preprocessing")
    merged.update(explicit)
    try:
        return ReferencePreprocessing(**merged)
    except ValueError as error:
        raise WorkerError(
            "invalid_reference",
            "reference.preprocessingの値が不正です。",
            stage="validation",
            reason="invalid_preprocessing_value",
        ) from error


def _invalid_preprocessing_value(name: str) -> WorkerError:
    return WorkerError(
        "invalid_reference",
        f"{name}の型が不正です。",
        stage="validation",
        reason="invalid_preprocessing_value",
    )


def _preprocessing_from_settings(settings: VoiceGenerationSettings) -> ReferencePreprocessing:
    return ReferencePreprocessing(
        ref_normalize_db=settings.ref_normalize_db,
        ref_ensure_max=settings.ref_ensure_max,
        max_ref_seconds=settings.max_ref_seconds,
    )


def _validate_prepared_voice_id(value: Any) -> str:
    if not isinstance(value, str) or _HANDLE_PATTERN.fullmatch(value) is None:
        raise WorkerError(
            "invalid_handle",
            "prepared_voice_idの形式が正しくありません。",
            stage="handle_resolution",
            reason="malformed",
        )
    return value


def _validate_reference_revision(value: Any) -> str:
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise WorkerError(
            "invalid_reference",
            "reference_revisionの形式が正しくありません。",
            stage="fallback",
            reason="invalid_revision",
        )
    return value


def _validate_no_ref(value: Any) -> bool:
    if not isinstance(value, bool):
        raise WorkerError(
            "invalid_request",
            "no_refはbooleanにしてください。",
            stage="validation",
            reason="invalid_no_ref",
        )
    return value


def _contradictory(reason: str) -> WorkerError:
    return WorkerError(
        "invalid_request",
        "矛盾する音声conditioning指定です。",
        stage="validation",
        reason=reason,
    )


def _reject_reference_output_collisions(references: Iterable[Path], output: Path) -> None:
    for reference in references:
        if reference == output:
            raise _reference_output_collision()
        try:
            reference_stat = reference.stat()
            output_stat = output.stat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise WorkerError(
                "invalid_request",
                "参照音声と出力先のfile identityを確認できません。",
                stage="validation",
                reason="reference_output_identity_unavailable",
            ) from error
        if os.path.samestat(reference_stat, output_stat):
            raise _reference_output_collision()


def _reference_output_collision() -> WorkerError:
    return WorkerError(
        "invalid_request",
        "参照音声と出力先には別のfileを指定してください。",
        stage="validation",
        reason="reference_is_output",
    )


def _validate_output_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("invalid_request", "output_pathを指定してください。")

    path = _resolve_protocol_path(value)
    if path.exists() and path.is_dir():
        raise WorkerError("invalid_request", "output_pathはfile pathを指定してください。")
    if not path.name:
        raise WorkerError("invalid_request", "output_pathはfile pathを指定してください。")
    return path


def _validate_output_dir(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("invalid_request", "output_dirを指定してください。")

    path = _resolve_protocol_path(value)
    if path.exists() and not path.is_dir():
        raise WorkerError("invalid_request", "output_dirはdirectory pathを指定してください。")
    return path


def _resolve_protocol_path(value: str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


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
        elif raw_value is None and key == "ref_normalize_db":
            coerced[key] = None
        else:
            if not isinstance(raw_value, expected) or isinstance(raw_value, bool):
                raise WorkerError("invalid_settings", f"{key}はnumberにしてください。")
            coerced[key] = float(raw_value)

    settings = VoiceGenerationSettings(**coerced)
    try:
        _preprocessing_from_settings(settings)
    except ValueError as error:
        raise WorkerError(
            "invalid_settings",
            "参照音声preprocessing settingsの値が不正です。",
        ) from error
    return settings


def _redirect_stdout_to_stderr(
    error_stream: TextIO,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    with contextlib.redirect_stdout(error_stream):
        return func(*args, **kwargs)


def _error_response(
    request_id: Any,
    code: str,
    message: str,
    *,
    stage: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if stage is not None:
        error["stage"] = stage
    if reason is not None:
        error["reason"] = reason
    return {"id": request_id, "ok": False, "error": error}


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
    with contextlib.suppress(Exception):
        traceback.print_exc(file=error_stream)


if __name__ == "__main__":
    main()
