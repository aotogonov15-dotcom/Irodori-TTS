from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import torch

from irodori_tts.inference_runtime import PreparedReferenceConditioning
from irodori_tts.prepared_voice_cache import (
    CacheClosedError,
    PreparedVoiceCache,
    ReferencePreprocessing,
    ReferenceSnapshotError,
    StalePreparedVoiceHandle,
    estimate_prepared_bytes,
    reference_revision,
    snapshot_reference_file,
)


def _prepared(tokens: int = 2, dimension: int = 3) -> PreparedReferenceConditioning:
    return PreparedReferenceConditioning(
        speaker_state=torch.zeros((1, tokens, dimension), dtype=torch.float32),
        speaker_mask=torch.ones((1, tokens), dtype=torch.bool),
        _lora_adapter=None,
    )


class ReferenceRevisionTest(unittest.TestCase):
    def test_content_and_effective_preprocessing_define_revision(self) -> None:
        default = ReferencePreprocessing()
        changed = ReferencePreprocessing(ref_normalize_db=-12.0)

        self.assertEqual(
            reference_revision([b"same"], default, kind="audio_file"),
            reference_revision([b"same"], default, kind="audio_file"),
        )
        self.assertNotEqual(
            reference_revision([b"same"], default, kind="audio_file"),
            reference_revision([b"different"], default, kind="audio_file"),
        )
        self.assertNotEqual(
            reference_revision([b"same"], default, kind="audio_file"),
            reference_revision([b"same"], changed, kind="audio_file"),
        )

    def test_nonpositive_max_seconds_normalizes_to_same_effective_identity(self) -> None:
        negative = ReferencePreprocessing(max_ref_seconds=-1)
        zero = ReferencePreprocessing(max_ref_seconds=0)
        explicit_none = ReferencePreprocessing(max_ref_seconds=None)

        self.assertEqual(negative, zero)
        self.assertEqual(zero, explicit_none)
        self.assertEqual(
            reference_revision([b"audio"], negative, kind="audio_file"),
            reference_revision([b"audio"], explicit_none, kind="audio_file"),
        )

    def test_ensure_max_is_canonical_when_loudness_normalization_makes_it_ineffective(
        self,
    ) -> None:
        enabled = ReferencePreprocessing(ref_normalize_db=-16.0, ref_ensure_max=True)
        disabled = ReferencePreprocessing(ref_normalize_db=-16.0, ref_ensure_max=False)

        self.assertEqual(enabled, disabled)
        self.assertFalse(enabled.ref_ensure_max)
        self.assertEqual(
            reference_revision([b"audio"], enabled, kind="audio_file"),
            reference_revision([b"audio"], disabled, kind="audio_file"),
        )

    def test_ensure_max_remains_part_of_identity_without_loudness_normalization(self) -> None:
        enabled = ReferencePreprocessing(ref_normalize_db=None, ref_ensure_max=True)
        disabled = ReferencePreprocessing(ref_normalize_db=None, ref_ensure_max=False)

        self.assertNotEqual(enabled, disabled)
        self.assertNotEqual(
            reference_revision([b"audio"], enabled, kind="audio_file"),
            reference_revision([b"audio"], disabled, kind="audio_file"),
        )

    def test_ordered_content_hashes_preserve_future_multi_reference_order(self) -> None:
        preprocessing = ReferencePreprocessing()
        self.assertNotEqual(
            reference_revision([b"a", b"b"], preprocessing, kind="audio_file"),
            reference_revision([b"b", b"a"], preprocessing, kind="audio_file"),
        )

    def test_snapshot_revision_tracks_bytes_not_path(self) -> None:
        preprocessing = ReferencePreprocessing()
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.wav"
            second = Path(temp_dir) / "second.wav"
            first.write_bytes(b"version-one")
            second.write_bytes(b"version-one")
            first_snapshot = snapshot_reference_file(first, preprocessing)
            second_snapshot = snapshot_reference_file(second, preprocessing)
            first.write_bytes(b"version-two")
            replaced_snapshot = snapshot_reference_file(first, preprocessing)

        self.assertEqual(first_snapshot.revision, second_snapshot.revision)
        self.assertNotEqual(first_snapshot.revision, replaced_snapshot.revision)
        self.assertEqual(first_snapshot.data, b"version-one")

    def test_snapshot_enforces_explicit_input_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "large.wav"
            reference.write_bytes(b"12345")

            with self.assertRaises(ReferenceSnapshotError) as raised:
                snapshot_reference_file(
                    reference,
                    ReferencePreprocessing(),
                    max_bytes=4,
                )

        self.assertEqual(raised.exception.reason, "size_limit")


class PreparedVoiceCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.handles = (f"{index:032x}" for index in range(1, 100))

    def _cache(self, **kwargs) -> PreparedVoiceCache:
        return PreparedVoiceCache(
            "runtime-generation",
            handle_factory=lambda: next(self.handles),
            **kwargs,
        )

    def test_prepare_miss_hit_and_same_handle(self) -> None:
        cache = self._cache()
        key = cache.cache_key("a" * 64)
        calls = 0

        def prepare() -> PreparedReferenceConditioning:
            nonlocal calls
            calls += 1
            return _prepared()

        miss = cache.get_or_prepare(key, prepare)
        hit = cache.get_or_prepare(key, prepare)

        self.assertEqual(calls, 1)
        self.assertFalse(miss.cache_hit)
        self.assertTrue(hit.cache_hit)
        self.assertEqual(miss.entry.prepared_voice_id, hit.entry.prepared_voice_id)
        self.assertIs(miss.prepared, hit.prepared)

    def test_different_reference_gets_different_handle(self) -> None:
        cache = self._cache()
        first = cache.get_or_prepare(cache.cache_key("a" * 64), _prepared)
        second = cache.get_or_prepare(cache.cache_key("b" * 64), _prepared)

        self.assertNotEqual(first.entry.prepared_voice_id, second.entry.prepared_voice_id)

    def test_lru_hit_moves_entry_and_count_eviction_removes_both_indexes(self) -> None:
        cache = self._cache(max_entries=2)
        first = cache.get_or_prepare(cache.cache_key("a" * 64), _prepared)
        second = cache.get_or_prepare(cache.cache_key("b" * 64), _prepared)
        cache.get_or_prepare(cache.cache_key("a" * 64), _prepared)
        third = cache.get_or_prepare(cache.cache_key("c" * 64), _prepared)

        self.assertEqual(third.evicted_count, 1)
        self.assertEqual(cache.entry_count, 2)
        self.assertIs(cache.acquire_handle(first.entry.prepared_voice_id).prepared, first.prepared)
        with self.assertRaises(StalePreparedVoiceHandle):
            cache.acquire_handle(second.entry.prepared_voice_id)
        replacement = cache.get_or_prepare(cache.cache_key("b" * 64), _prepared)
        self.assertNotEqual(replacement.entry.prepared_voice_id, second.entry.prepared_voice_id)

    def test_byte_limit_and_combined_limits_are_enforced(self) -> None:
        item = _prepared()
        item_bytes = estimate_prepared_bytes(item)
        cache = self._cache(max_entries=2, max_bytes=item_bytes * 2)

        first = cache.get_or_prepare(cache.cache_key("a" * 64), lambda: item)
        second = cache.get_or_prepare(cache.cache_key("b" * 64), _prepared)
        third = cache.get_or_prepare(cache.cache_key("c" * 64), _prepared)

        self.assertEqual(cache.entry_count, 2)
        self.assertLessEqual(cache.estimated_bytes, item_bytes * 2)
        with self.assertRaises(StalePreparedVoiceHandle):
            cache.acquire_handle(first.entry.prepared_voice_id)
        self.assertIsNotNone(second.entry)
        self.assertIsNotNone(third.entry)

    def test_oversized_prepared_is_returned_once_but_not_cached(self) -> None:
        item = _prepared(tokens=4, dimension=4)
        cache = self._cache(max_bytes=estimate_prepared_bytes(item) - 1)
        calls = 0

        def prepare() -> PreparedReferenceConditioning:
            nonlocal calls
            calls += 1
            return item

        result = cache.get_or_prepare(cache.cache_key("a" * 64), prepare)

        self.assertIs(result.prepared, item)
        self.assertIsNone(result.entry)
        self.assertEqual(calls, 1)
        self.assertEqual(cache.entry_count, 0)

    def test_active_strong_reference_survives_eviction(self) -> None:
        cache = self._cache(max_entries=1)
        first = cache.get_or_prepare(cache.cache_key("a" * 64), _prepared)
        active = cache.acquire_handle(first.entry.prepared_voice_id).prepared
        original_state = active.speaker_state

        cache.get_or_prepare(cache.cache_key("b" * 64), _prepared)

        with self.assertRaises(StalePreparedVoiceHandle):
            cache.acquire_handle(first.entry.prepared_voice_id)
        self.assertIs(active.speaker_state, original_state)

    def test_shared_tensor_storage_is_not_double_counted(self) -> None:
        storage = torch.zeros(48, dtype=torch.uint8)
        state = storage.view(torch.float32).view(1, 3, 4)
        mask = storage[:3].view(torch.bool).view(1, 3)
        mask.fill_(True)
        prepared = PreparedReferenceConditioning(
            speaker_state=state,
            speaker_mask=mask,
            _lora_adapter=None,
        )

        self.assertEqual(
            estimate_prepared_bytes(prepared),
            storage.untyped_storage().nbytes(),
        )

    def test_same_key_concurrent_prepare_is_single_flight(self) -> None:
        cache = self._cache()
        key = cache.cache_key("a" * 64)
        leader_started = threading.Event()
        release_leader = threading.Event()
        waiter_joined = threading.Event()
        calls = 0
        results = []

        class _ObservedFuture(Future):
            def result(self, timeout=None):
                waiter_joined.set()
                return super().result(timeout=timeout)

        def prepare() -> PreparedReferenceConditioning:
            nonlocal calls
            calls += 1
            leader_started.set()
            self.assertTrue(release_leader.wait(timeout=5))
            return _prepared()

        def participant() -> None:
            results.append(cache.get_or_prepare(key, prepare))

        with patch("irodori_tts.prepared_voice_cache.Future", _ObservedFuture):
            leader = threading.Thread(target=participant)
            waiter = threading.Thread(target=participant)
            leader.start()
            self.assertTrue(leader_started.wait(timeout=5))
            waiter.start()
            self.assertTrue(waiter_joined.wait(timeout=5))
            self.assertTrue(leader.is_alive())
            self.assertEqual(cache.in_flight_count, 1)
            release_leader.set()
            leader.join(timeout=5)
            waiter.join(timeout=5)

        self.assertFalse(leader.is_alive())
        self.assertFalse(waiter.is_alive())
        self.assertEqual(calls, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0].entry.prepared_voice_id,
            results[1].entry.prepared_voice_id,
        )

    def test_leader_failure_propagates_and_later_retry_succeeds(self) -> None:
        cache = self._cache()
        key = cache.cache_key("a" * 64)
        leader_started = threading.Event()
        release_leader = threading.Event()
        waiter_joined = threading.Event()
        errors = []

        class _ObservedFuture(Future):
            def result(self, timeout=None):
                waiter_joined.set()
                return super().result(timeout=timeout)

        def fail() -> PreparedReferenceConditioning:
            leader_started.set()
            self.assertTrue(release_leader.wait(timeout=5))
            raise ValueError("prepare failed")

        def participant() -> None:
            try:
                cache.get_or_prepare(key, fail)
            except Exception as error:  # noqa: BLE001 - asserting shared propagation
                errors.append(error)

        with patch("irodori_tts.prepared_voice_cache.Future", _ObservedFuture):
            leader = threading.Thread(target=participant)
            waiter = threading.Thread(target=participant)
            leader.start()
            self.assertTrue(leader_started.wait(timeout=5))
            waiter.start()
            self.assertTrue(waiter_joined.wait(timeout=5))
            self.assertTrue(leader.is_alive())
            self.assertEqual(cache.in_flight_count, 1)
            release_leader.set()
            leader.join(timeout=5)
            waiter.join(timeout=5)

        self.assertFalse(leader.is_alive())
        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(str(error) == "prepare failed" for error in errors))
        self.assertEqual(cache.in_flight_count, 0)
        self.assertIsNotNone(cache.get_or_prepare(key, _prepared).entry)

    def test_cancelled_waiter_does_not_cancel_leader_or_corrupt_result(self) -> None:
        cache = self._cache()
        key = cache.cache_key("a" * 64)
        leader_started = threading.Event()
        release_leader = threading.Event()
        leader_results = []

        def prepare() -> PreparedReferenceConditioning:
            leader_started.set()
            self.assertTrue(release_leader.wait(timeout=5))
            return _prepared()

        leader = threading.Thread(
            target=lambda: leader_results.append(cache.get_or_prepare(key, prepare))
        )
        leader.start()
        self.assertTrue(leader_started.wait(timeout=5))
        cancelled = threading.Event()
        cancelled.set()

        with self.assertRaises(FutureCancelledError):
            cache.get_or_prepare(
                key,
                prepare,
                waiter_cancelled=cancelled.is_set,
            )

        release_leader.set()
        leader.join(timeout=5)
        self.assertEqual(len(leader_results), 1)
        self.assertIs(
            cache.acquire_handle(leader_results[0].entry.prepared_voice_id).prepared,
            leader_results[0].prepared,
        )

    def test_close_during_prepare_prevents_publication(self) -> None:
        cache = self._cache()
        key = cache.cache_key("a" * 64)
        prepare_started = threading.Event()
        release_prepare = threading.Event()
        errors = []

        def prepare() -> PreparedReferenceConditioning:
            prepare_started.set()
            self.assertTrue(release_prepare.wait(timeout=5))
            return _prepared()

        def leader() -> None:
            try:
                cache.get_or_prepare(key, prepare)
            except Exception as error:  # noqa: BLE001 - asserted below
                errors.append(error)

        thread = threading.Thread(target=leader)
        thread.start()
        self.assertTrue(prepare_started.wait(timeout=5))
        cache.close()
        release_prepare.set()
        thread.join(timeout=5)

        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache.in_flight_count, 0)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], CacheClosedError)


if __name__ == "__main__":
    unittest.main()
