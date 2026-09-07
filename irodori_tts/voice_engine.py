from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)


# =========================================================
# モデル設定
# =========================================================

MODEL_REPO = "Aratako/Irodori-TTS-500M-v3"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"

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
    ref_normalize_db: float = -16.0
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
        reference_audio: str | Path,
        output_dir: str | Path,
    ) -> None:
        self.reference_audio = Path(reference_audio)
        self.output_dir = Path(output_dir)

        self._runtime: InferenceRuntime | None = None

    @property
    def is_loaded(self) -> bool:
        """モデルが読み込まれているかを返す。"""

        return self._runtime is not None

    def load(self) -> None:
        """モデルとCodecを読み込む。既に読み込み済みなら何もしない。"""

        if self._runtime is not None:
            return

        if not self.reference_audio.is_file():
            raise FileNotFoundError(
                "参照音声が見つかりません。\n"
                f"確認する場所: {self.reference_audio}"
            )

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
                "音声生成エンジンが読み込まれていません。"
                "先にload()を実行してください。"
            )

        active_reference_audio = (
            Path(reference_audio)
            if reference_audio is not None
            else self.reference_audio
        )

        if not active_reference_audio.is_file():
            raise FileNotFoundError(
                "参照音声ファイルが見つかりません。\n"
                f"確認する場所: {active_reference_audio}"
            )

        result = self._runtime.synthesize(
            SamplingRequest(
                text=cleaned_text,
                ref_wav=str(active_reference_audio),

                num_steps=active_settings.num_steps,
                t_schedule_mode=active_settings.t_schedule_mode,
                sway_coeff=active_settings.sway_coeff,

                cfg_guidance_mode=active_settings.cfg_guidance_mode,
                cfg_scale_text=active_settings.cfg_scale_text,
                cfg_scale_speaker=active_settings.cfg_scale_speaker,

                duration_scale=active_settings.duration_scale,
                seed=active_settings.seed,

                num_candidates=active_settings.num_candidates,
                decode_mode=active_settings.decode_mode,

                ref_normalize_db=active_settings.ref_normalize_db,
                ref_ensure_max=active_settings.ref_ensure_max,
                max_ref_seconds=active_settings.max_ref_seconds,
            ),
            log_fn=None,
        )

        active_output_path = (
            Path(output_path)
            if output_path is not None
            else self._create_output_path()
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
