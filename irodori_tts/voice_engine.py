from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    PreparedReferenceConditioning,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)

# =========================================================
# モデル設定
# =========================================================

MODEL_REPO = "Aratako/Irodori-TTS-500M-v3"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
CHARACTER_BACKEND = "irodori"
CHARACTER_BASE_MODEL_ID = MODEL_REPO

MODEL_DEVICE = "cuda"
MODEL_PRECISION = "bf16"

CODEC_DEVICE = "cuda"
CODEC_PRECISION = "bf16"


@dataclass(frozen=True)
class VoiceGenerationSettings:
    num_steps: int = 16
    t_schedule_mode: str = "sway"
    sway_coeff: float = -1.0
    cfg_guidance_mode: str = "independent"
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 7.0
    duration_scale: float = 1.0
    seed: int = 1234
    num_candidates: int = 1
    decode_mode: str = "sequential"
    ref_normalize_db: float | None = -16.0
    ref_ensure_max: bool = True
    max_ref_seconds: float = 30.0


@dataclass(frozen=True)
class VoiceGenerationResult:
    """音声生成結果をまとめて返すためのデータ。"""

    output_path: Path
    used_seed: int
    generation_seconds: float


class VoiceEngine:
    """Irodori-TTSのモデル読込と音声生成を担当するクラス。"""

    def __init__(
        self,
        reference_audio: str | Path | None,
        output_dir: str | Path,
    ) -> None:
        self.reference_audio = Path(reference_audio) if reference_audio is not None else None
        self.output_dir = Path(output_dir)

        self._runtime: InferenceRuntime | None = None

    @property
    def is_loaded(self) -> bool:
        """モデルが読み込まれているかを返す。"""

        return self._runtime is not None

    @property
    def runtime_generation(self) -> str:
        if self._runtime is None:
            raise RuntimeError("音声生成エンジンが読み込まれていません。")
        return self._runtime.runtime_generation

    @property
    def character_state(self) -> str:
        return self._require_runtime().lifecycle_state.value

    def load(self) -> None:
        """参照音声に依存せずモデルとCodecを読み込む。読み込み済みなら何もしない。"""

        if self._runtime is not None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="model.safetensors",
        )

        self._runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint_path,
                model_device=MODEL_DEVICE,
                model_precision=MODEL_PRECISION,
                codec_repo=CODEC_REPO,
                codec_device=CODEC_DEVICE,
                codec_precision=CODEC_PRECISION,
                codec_deterministic_encode=True,
                codec_deterministic_decode=True,
                compile_model=False,
                compile_dynamic=False,
            )
        )

    def generate(
        self,
        text: str,
        reference_audio: str | Path | None = None,
        settings: VoiceGenerationSettings | None = None,
        output_path: str | Path | None = None,
    ) -> VoiceGenerationResult:
        """入力された文章から音声を生成して保存する。"""

        cleaned_text = text.strip()
        active_settings = settings or VoiceGenerationSettings()

        if not cleaned_text:
            raise ValueError("文章が入力されていません。")

        if self._runtime is None:
            raise RuntimeError(
                "音声生成エンジンが読み込まれていません。先にload()を実行してください。"
            )

        active_reference_audio = (
            Path(reference_audio) if reference_audio is not None else self.reference_audio
        )

        if active_reference_audio is None or not active_reference_audio.is_file():
            raise FileNotFoundError(
                f"参照音声ファイルが見つかりません。\n確認する場所: {active_reference_audio}"
            )

        result = self._runtime.synthesize(
            self._sampling_request(
                cleaned_text,
                active_settings,
                ref_wav=str(active_reference_audio),
            ),
            log_fn=None,
        )

        return self._save_result(result, output_path)

    def prepare_reference_conditioning(
        self,
        reference_snapshot: bytes,
        *,
        ref_normalize_db: float | None,
        ref_ensure_max: bool,
        max_ref_seconds: float | None,
    ) -> PreparedReferenceConditioning:
        """Prepare from the exact immutable bytes hashed by the worker."""
        runtime = self._require_runtime()
        return runtime.prepare_reference_conditioning(
            ref_wav_bytes=reference_snapshot,
            ref_normalize_db=ref_normalize_db,
            ref_ensure_max=ref_ensure_max,
            max_ref_seconds=max_ref_seconds,
        )

    def claim_character(self, lora_adapter: str | Path | None = None) -> None:
        """Permanently finalize this runtime for one base or LoRA character."""
        runtime = self._require_runtime()
        runtime.finalize_character(
            None if lora_adapter is None else str(lora_adapter),
        )

    def generate_with_prepared(
        self,
        text: str,
        prepared_reference: PreparedReferenceConditioning,
        settings: VoiceGenerationSettings | None = None,
        output_path: str | Path | None = None,
    ) -> VoiceGenerationResult:
        """Generate from worker-owned Prepared state without injecting a default WAV."""
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("文章が入力されていません。")
        runtime = self._require_runtime()
        active_settings = settings or VoiceGenerationSettings()
        result = runtime.synthesize(
            self._sampling_request(cleaned_text, active_settings),
            log_fn=None,
            prepared_reference=prepared_reference,
        )
        return self._save_result(result, output_path)

    def generate_no_ref(
        self,
        text: str,
        settings: VoiceGenerationSettings | None = None,
        output_path: str | Path | None = None,
    ) -> VoiceGenerationResult:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("文章が入力されていません。")
        runtime = self._require_runtime()
        active_settings = settings or VoiceGenerationSettings()
        result = runtime.synthesize(
            self._sampling_request(cleaned_text, active_settings, no_ref=True),
            log_fn=None,
        )
        return self._save_result(result, output_path)

    def close(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            runtime.unload()

    def _require_runtime(self) -> InferenceRuntime:
        if self._runtime is None:
            raise RuntimeError(
                "音声生成エンジンが読み込まれていません。先にload()を実行してください。"
            )
        return self._runtime

    @staticmethod
    def _sampling_request(
        text: str,
        settings: VoiceGenerationSettings,
        *,
        ref_wav: str | None = None,
        no_ref: bool = False,
    ) -> SamplingRequest:
        return SamplingRequest(
            text=text,
            ref_wav=ref_wav,
            no_ref=no_ref,
            num_steps=settings.num_steps,
            t_schedule_mode=settings.t_schedule_mode,
            sway_coeff=settings.sway_coeff,
            cfg_guidance_mode=settings.cfg_guidance_mode,
            cfg_scale_text=settings.cfg_scale_text,
            cfg_scale_speaker=settings.cfg_scale_speaker,
            duration_scale=settings.duration_scale,
            seed=settings.seed,
            num_candidates=settings.num_candidates,
            decode_mode=settings.decode_mode,
            ref_normalize_db=settings.ref_normalize_db,
            ref_ensure_max=settings.ref_ensure_max,
            max_ref_seconds=settings.max_ref_seconds,
        )

    def _save_result(
        self,
        result: object,
        output_path: str | Path | None,
    ) -> VoiceGenerationResult:

        active_output_path = (
            Path(output_path) if output_path is not None else self._create_output_path()
        )

        saved_path = save_wav(
            active_output_path,
            result.audio,
            result.sample_rate,
        )

        return VoiceGenerationResult(
            output_path=Path(saved_path).resolve(),
            used_seed=result.used_seed,
            generation_seconds=result.total_to_decode,
        )

    def _create_output_path(self) -> Path:
        """重複しない日時入りの出力ファイル名を作る。"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.output_dir / f"voice_{timestamp}.wav"
