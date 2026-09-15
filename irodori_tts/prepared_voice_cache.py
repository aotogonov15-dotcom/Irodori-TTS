from __future__ import annotations

import hashlib
import json
import math
import secrets
import stat
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch

from .inference_runtime import PreparedReferenceConditioning

PREPARATION_SCHEMA_REVISION = 1
DEFAULT_MAX_CACHE_ENTRIES = 64
DEFAULT_MAX_CACHE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_REFERENCE_BYTES = 64 * 1024 * 1024


class CacheClosedError(RuntimeError):
    """The cache generation is no longer available."""


class CacheInvariantError(RuntimeError):
    """The cache indexes or runtime namespace disagree."""


class StalePreparedVoiceHandle(KeyError):
    """A syntactically valid handle is not resident."""


class ReferenceSnapshotError(ValueError):
    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ReferencePreprocessing:
    ref_normalize_db: float | None = -16.0
    ref_ensure_max: bool = True
    max_ref_seconds: float | None = 30.0

    def __post_init__(self) -> None:
        normalize_db = self.ref_normalize_db
        max_seconds = self.max_ref_seconds
        if normalize_db is not None:
            normalized = float(normalize_db)
            if not math.isfinite(normalized):
                raise ValueError("ref_normalize_db must be finite or null.")
            object.__setattr__(self, "ref_normalize_db", 0.0 if normalized == 0 else normalized)
        if not isinstance(self.ref_ensure_max, bool):
            raise ValueError("ref_ensure_max must be boolean.")
        if normalize_db is not None:
            object.__setattr__(self, "ref_ensure_max", False)
        if max_seconds is not None:
            normalized_max = float(max_seconds)
            if not math.isfinite(normalized_max):
                raise ValueError("max_ref_seconds must be finite or null.")
            object.__setattr__(
                self,
                "max_ref_seconds",
                None if normalized_max <= 0 else normalized_max,
            )

    def revision_payload(self) -> dict[str, float | bool | None]:
        return {
            "max_ref_seconds": self.max_ref_seconds,
            "ref_ensure_max": self.ref_ensure_max,
            "ref_normalize_db": self.ref_normalize_db,
        }


@dataclass(frozen=True)
class ReferenceSnapshot:
    data: bytes
    revision: str
    preprocessing: ReferencePreprocessing


@dataclass(frozen=True)
class PreparedVoiceCacheKey:
    runtime_generation: str
    preparation_schema_revision: int
    reference_revision: str


@dataclass(frozen=True)
class PreparedVoiceCacheEntry:
    key: PreparedVoiceCacheKey
    prepared: PreparedReferenceConditioning
    prepared_voice_id: str
    estimated_bytes: int


@dataclass(frozen=True)
class PreparedVoiceAcquisition:
    prepared: PreparedReferenceConditioning
    entry: PreparedVoiceCacheEntry | None
    cache_hit: bool
    estimated_bytes: int
    coalesced_waiters: int = 0
    evicted_count: int = 0

    @property
    def cached(self) -> bool:
        return self.entry is not None


@dataclass
class _InFlightPreparation:
    future: Future[PreparedVoiceAcquisition]
    waiters: int = 0


def snapshot_reference_file(
    path: Path,
    preprocessing: ReferencePreprocessing,
    *,
    max_bytes: int = DEFAULT_MAX_REFERENCE_BYTES,
) -> ReferenceSnapshot:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    try:
        file_stat = path.stat()
    except OSError as error:
        raise ReferenceSnapshotError(
            "Reference audio file is unavailable.",
            reason="not_found",
        ) from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise ReferenceSnapshotError(
            "Reference audio must be a regular local file.",
            reason="not_regular_file",
        )
    if file_stat.st_size > max_bytes:
        raise ReferenceSnapshotError(
            f"Reference audio exceeds the {max_bytes}-byte input limit.",
            reason="size_limit",
        )
    try:
        with path.open("rb") as stream:
            data = stream.read(max_bytes + 1)
    except OSError as error:
        raise ReferenceSnapshotError(
            "Reference audio could not be read.",
            reason="read_failed",
        ) from error
    if len(data) > max_bytes:
        raise ReferenceSnapshotError(
            f"Reference audio exceeds the {max_bytes}-byte input limit.",
            reason="size_limit",
        )
    if not data:
        raise ReferenceSnapshotError(
            "Reference audio is empty.",
            reason="empty",
        )
    return ReferenceSnapshot(
        data=data,
        revision=reference_revision([data], preprocessing, kind="audio_file"),
        preprocessing=preprocessing,
    )


def reference_revision(
    ordered_contents: list[bytes],
    preprocessing: ReferencePreprocessing,
    *,
    kind: str,
) -> str:
    payload = {
        "content_sha256": [hashlib.sha256(content).hexdigest() for content in ordered_contents],
        "kind": kind,
        "preparation_schema_revision": PREPARATION_SCHEMA_REVISION,
        "preprocessing": preprocessing.revision_payload(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def estimate_prepared_bytes(prepared: PreparedReferenceConditioning) -> int:
    """Estimate retained tensor storage without double-counting shared storage."""
    values: list[Any]
    if is_dataclass(prepared):
        values = [getattr(prepared, item.name) for item in fields(prepared)]
    else:  # pragma: no cover - permits lightweight compatible test doubles
        values = list(vars(prepared).values())

    total = 0
    seen_storages: set[tuple[str, int | None, int, int]] = set()
    for value in values:
        if not isinstance(value, torch.Tensor):
            continue
        fallback_bytes = int(value.numel()) * int(value.element_size())
        try:
            storage = value.untyped_storage()
            storage_bytes = int(storage.nbytes())
            storage_key = (
                value.device.type,
                value.device.index,
                int(storage.data_ptr()),
                storage_bytes,
            )
        except (AttributeError, RuntimeError):  # pragma: no cover - unusual tensor backends
            storage_key = (
                value.device.type,
                value.device.index,
                int(value.data_ptr()),
                fallback_bytes,
            )
            storage_bytes = fallback_bytes
        if storage_key in seen_storages:
            continue
        seen_storages.add(storage_key)
        total += max(fallback_bytes, storage_bytes)
    return total


class PreparedVoiceCache:
    """Bounded, generation-scoped worker ownership for Prepared state."""

    def __init__(
        self,
        runtime_generation: str,
        *,
        max_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
        max_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        handle_factory: Callable[[], str] | None = None,
    ) -> None:
        if not runtime_generation:
            raise ValueError("runtime_generation must be non-empty.")
        if max_entries <= 0 or max_bytes <= 0:
            raise ValueError("Cache limits must be positive.")
        self.runtime_generation = runtime_generation
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._handle_factory = handle_factory or (lambda: secrets.token_hex(16))
        self._lock = threading.Lock()
        self._entries_by_key: dict[PreparedVoiceCacheKey, PreparedVoiceCacheEntry] = {}
        self._entries_by_handle: dict[str, PreparedVoiceCacheEntry] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._in_flight: dict[PreparedVoiceCacheKey, _InFlightPreparation] = {}
        self._estimated_bytes = 0
        self._accepting = True

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries_by_handle)

    @property
    def estimated_bytes(self) -> int:
        with self._lock:
            return self._estimated_bytes

    @property
    def in_flight_count(self) -> int:
        with self._lock:
            return len(self._in_flight)

    def cache_key(self, reference_revision_value: str) -> PreparedVoiceCacheKey:
        return PreparedVoiceCacheKey(
            runtime_generation=self.runtime_generation,
            preparation_schema_revision=PREPARATION_SCHEMA_REVISION,
            reference_revision=reference_revision_value,
        )

    def acquire_handle(self, prepared_voice_id: str) -> PreparedVoiceCacheEntry:
        with self._lock:
            self._raise_if_closed()
            entry = self._entries_by_handle.get(prepared_voice_id)
            if entry is None:
                raise StalePreparedVoiceHandle(prepared_voice_id)
            if entry.key.runtime_generation != self.runtime_generation:
                raise CacheInvariantError("Resident handle belongs to another runtime generation.")
            if self._entries_by_key.get(entry.key) is not entry:
                raise CacheInvariantError("Prepared voice cache indexes disagree.")
            self._touch(entry)
            return entry

    def get_or_prepare(
        self,
        key: PreparedVoiceCacheKey,
        prepare: Callable[[], PreparedReferenceConditioning],
        *,
        waiter_cancelled: Callable[[], bool] | None = None,
    ) -> PreparedVoiceAcquisition:
        if key.runtime_generation != self.runtime_generation:
            raise CacheInvariantError("Cache key belongs to another runtime generation.")
        if waiter_cancelled is not None and waiter_cancelled():
            raise FutureCancelledError("Prepared voice waiter was cancelled.")

        with self._lock:
            self._raise_if_closed()
            cached = self._entries_by_key.get(key)
            if cached is not None:
                self._touch(cached)
                return PreparedVoiceAcquisition(
                    prepared=cached.prepared,
                    entry=cached,
                    cache_hit=True,
                    estimated_bytes=cached.estimated_bytes,
                )
            flight = self._in_flight.get(key)
            if flight is None:
                flight = _InFlightPreparation(future=Future())
                self._in_flight[key] = flight
                is_leader = True
            else:
                flight.waiters += 1
                is_leader = False

        if not is_leader:
            if waiter_cancelled is not None and waiter_cancelled():
                with self._lock:
                    if self._in_flight.get(key) is flight:
                        flight.waiters -= 1
                raise FutureCancelledError("Prepared voice waiter was cancelled.")
            result = flight.future.result()
            return PreparedVoiceAcquisition(
                prepared=result.prepared,
                entry=result.entry,
                cache_hit=False,
                estimated_bytes=result.estimated_bytes,
                coalesced_waiters=result.coalesced_waiters,
                evicted_count=result.evicted_count,
            )

        try:
            prepared = prepare()
            estimated_bytes = estimate_prepared_bytes(prepared)
            with self._lock:
                self._in_flight.pop(key, None)
                if not self._accepting:
                    raise CacheClosedError("Prepared voice cache was closed during preparation.")
                entry = None
                evicted_count = 0
                if estimated_bytes <= self.max_bytes:
                    entry, evicted_count = self._insert(key, prepared, estimated_bytes)
                result = PreparedVoiceAcquisition(
                    prepared=prepared,
                    entry=entry,
                    cache_hit=False,
                    estimated_bytes=estimated_bytes,
                    coalesced_waiters=flight.waiters,
                    evicted_count=evicted_count,
                )
                if not flight.future.done():
                    flight.future.set_result(result)
            return result
        except BaseException as error:
            with self._lock:
                self._in_flight.pop(key, None)
            if not flight.future.done():
                flight.future.set_exception(error)
            raise

    def clear(self) -> None:
        with self._lock:
            self._clear_entries()

    def close(self) -> None:
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
            self._clear_entries()
            flights = list(self._in_flight.values())
            self._in_flight.clear()
        for flight in flights:
            if not flight.future.done():
                flight.future.set_exception(
                    CacheClosedError("Prepared voice cache generation is closed.")
                )

    def resident_handles_lru(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._lru)

    def _insert(
        self,
        key: PreparedVoiceCacheKey,
        prepared: PreparedReferenceConditioning,
        estimated_bytes: int,
    ) -> tuple[PreparedVoiceCacheEntry, int]:
        while True:
            handle = self._handle_factory()
            if handle not in self._entries_by_handle:
                break
        entry = PreparedVoiceCacheEntry(
            key=key,
            prepared=prepared,
            prepared_voice_id=handle,
            estimated_bytes=estimated_bytes,
        )
        self._entries_by_key[key] = entry
        self._entries_by_handle[handle] = entry
        self._lru[handle] = None
        self._estimated_bytes += estimated_bytes
        evicted_count = 0
        while (
            len(self._entries_by_handle) > self.max_entries
            or self._estimated_bytes > self.max_bytes
        ):
            oldest_handle, _ = self._lru.popitem(last=False)
            self._remove(oldest_handle)
            evicted_count += 1
        return entry, evicted_count

    def _remove(self, handle: str) -> None:
        entry = self._entries_by_handle.pop(handle)
        self._entries_by_key.pop(entry.key, None)
        self._lru.pop(handle, None)
        self._estimated_bytes -= entry.estimated_bytes

    def _touch(self, entry: PreparedVoiceCacheEntry) -> None:
        self._lru.move_to_end(entry.prepared_voice_id)

    def _clear_entries(self) -> None:
        self._entries_by_key.clear()
        self._entries_by_handle.clear()
        self._lru.clear()
        self._estimated_bytes = 0

    def _raise_if_closed(self) -> None:
        if not self._accepting:
            raise CacheClosedError("Prepared voice cache generation is closed.")
