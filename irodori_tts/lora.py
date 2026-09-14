from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .config import TrainConfig
from .model import TextToLatentRFDiT

LORA_TRAIN_CONFIG_FIELDS = (
    "lora_enabled",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_bias",
    "lora_target_modules",
    "lora_modules_to_save",
)

LORA_ADAPTER_CONFIG_NAME = "adapter_config.json"
LORA_ADAPTER_STATE_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
LORA_TRAINER_STATE_NAME = "trainer_state.pt"
LORA_METADATA_NAME = "irodori_lora_metadata.json"

CHARACTER_LORA_R = 16
CHARACTER_LORA_ALPHA = 32
CHARACTER_LORA_DROPOUT = 0.0
CHARACTER_LORA_MODULES_TO_SAVE = ("duration_predictor",)

_KNOWN_CHARACTER_LORA_CONFIG_FIELDS = {
    "alora_invocation_tokens",
    "alpha_pattern",
    "arrow_config",
    "auto_mapping",
    "base_model_name_or_path",
    "bias",
    "corda_config",
    "ensure_weight_tying",
    "eva_config",
    "exclude_modules",
    "fan_in_fan_out",
    "inference_mode",
    "init_lora_weights",
    "layer_replication",
    "layers_pattern",
    "layers_to_transform",
    "loftq_config",
    "lora_alpha",
    "lora_bias",
    "lora_dropout",
    "megatron_config",
    "megatron_core",
    "modules_to_save",
    "peft_type",
    "peft_version",
    "qalora_group_size",
    "r",
    "rank_pattern",
    "revision",
    "target_modules",
    "target_parameters",
    "task_type",
    "trainable_token_indices",
    "use_dora",
    "use_qalora",
    "use_rslora",
}


@dataclass(frozen=True)
class LoraAdapterPreflight:
    """Complete character-adapter input validated before PEFT mutation."""

    adapter_name: str
    peft_config: Any
    adapter_state: tuple[tuple[str, torch.Tensor], ...]
    expected_applied_state: tuple[tuple[str, torch.Tensor], ...]


LORA_TARGET_PRESETS: dict[str, str] = {
    "text_attn_mlp": (
        r"^text_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))$"
    ),
    "caption_attn_mlp": (
        r"^caption_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))$"
    ),
    "speaker_attn_mlp": (
        r"^(speaker_encoder\.in_proj"
        r"|speaker_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3)))$"
    ),
    "diffusion_attn": (
        r"^blocks\.\d+\.attention\."
        r"(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)$"
    ),
    "diffusion_attn_mlp": (
        r"^blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)"
        r"|mlp\.(w1|w2|w3))$"
    ),
    "all_attn": (
        r"^(text_encoder\.blocks\.\d+\.attention\.(wq|wk|wv|wo|gate)"
        r"|caption_encoder\.blocks\.\d+\.attention\.(wq|wk|wv|wo|gate)"
        r"|speaker_encoder\.blocks\.\d+\.attention\.(wq|wk|wv|wo|gate)"
        r"|blocks\.\d+\.attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate))$"
    ),
    "diffusion_full": (
        r"^(cond_module\.(0|2|4)"
        r"|in_proj"
        r"|out_proj"
        r"|blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)"
        r"|mlp\.(w1|w2|w3)"
        r"|attention_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)"
        r"|mlp_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)))$"
    ),
    "adaln": (
        r"^blocks\.\d+\."
        r"(attention_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)"
        r"|mlp_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up))$"
    ),
    "conditioning": (
        r"^(cond_module\.(0|2|4)"
        r"|speaker_encoder\.in_proj"
        r"|blocks\.\d+\.attention\.(wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption))$"
    ),
    "all_attn_mlp": (
        r"^(text_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|caption_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|speaker_encoder\.in_proj"
        r"|speaker_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)"
        r"|mlp\.(w1|w2|w3)))$"
    ),
    "all_linear": (
        r"^(speaker_encoder\.in_proj"
        r"|cond_module\.(0|2|4)"
        r"|in_proj"
        r"|out_proj"
        r"|text_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|caption_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|speaker_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate|wo)"
        r"|mlp\.(w1|w2|w3)"
        r"|attention_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)"
        r"|mlp_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)))$"
    ),
}


def _require_peft():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "LoRA fine-tuning requires `peft`. Install with `pip install peft` or `uv sync`."
        ) from exc
    return LoraConfig, PeftModel, get_peft_model


def _lookup_config_value(raw: TrainConfig | Mapping[str, Any] | None, field: str) -> Any:
    if raw is None:
        return getattr(TrainConfig(), field)
    if isinstance(raw, TrainConfig):
        return getattr(raw, field)
    if isinstance(raw, Mapping):
        if field in raw:
            return raw[field]
        return getattr(TrainConfig(), field)
    raise TypeError(f"Unsupported LoRA config source: {type(raw)!r}")


def train_config_uses_lora(raw: TrainConfig | Mapping[str, Any] | None) -> bool:
    return bool(_lookup_config_value(raw, "lora_enabled"))


def checkpoint_state_uses_lora(model_state: Mapping[str, torch.Tensor]) -> bool:
    return any(key.startswith("base_model.model.") or ".lora_" in key for key in model_state)


def resolve_lora_target_modules(spec: str | Sequence[str] | None) -> str | list[str]:
    if spec is None:
        spec = TrainConfig().lora_target_modules

    if isinstance(spec, str):
        value = spec.strip()
        if not value:
            raise ValueError("lora_target_modules must not be empty.")
        preset = LORA_TARGET_PRESETS.get(value)
        if preset is not None:
            return preset
        if "," in value:
            modules = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
            if not modules:
                raise ValueError(f"Invalid LoRA target_modules list: {spec!r}")
            return modules
        return value

    modules = [str(item).strip() for item in spec if str(item).strip()]
    if not modules:
        raise ValueError("LoRA target_modules sequence must not be empty.")
    return modules


def resolve_lora_modules_to_save(
    spec: str | Sequence[str] | None,
    *,
    use_duration_predictor: bool,
) -> list[str] | None:
    if spec is None:
        return None

    if isinstance(spec, str):
        value = spec.strip()
        if not value or value.lower() == "none":
            return None
        if value.lower() == "auto":
            if use_duration_predictor:
                return ["duration_predictor"]
            return None
        modules = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    else:
        modules = [str(item).strip() for item in spec if str(item).strip()]

    if not modules:
        return None
    return modules


def build_lora_config_kwargs(
    raw: TrainConfig | Mapping[str, Any],
    *,
    use_duration_predictor: bool = False,
) -> dict[str, Any]:
    bias = str(_lookup_config_value(raw, "lora_bias")).strip().lower()
    if bias not in {"none", "all", "lora_only"}:
        raise ValueError(f"Unsupported lora_bias={bias!r}. Expected one of: none, all, lora_only.")

    kwargs = {
        "r": int(_lookup_config_value(raw, "lora_r")),
        "lora_alpha": int(_lookup_config_value(raw, "lora_alpha")),
        "lora_dropout": float(_lookup_config_value(raw, "lora_dropout")),
        "bias": bias,
        "target_modules": resolve_lora_target_modules(
            _lookup_config_value(raw, "lora_target_modules")
        ),
    }
    modules_to_save = resolve_lora_modules_to_save(
        _lookup_config_value(raw, "lora_modules_to_save"),
        use_duration_predictor=use_duration_predictor,
    )
    if modules_to_save is not None:
        kwargs["modules_to_save"] = modules_to_save
    return kwargs


def apply_lora(
    model: TextToLatentRFDiT,
    raw: TrainConfig | Mapping[str, Any],
) -> torch.nn.Module:
    if not train_config_uses_lora(raw):
        return model

    lora_config_cls, _, get_peft_model = _require_peft()
    peft_model = get_peft_model(
        model,
        lora_config_cls(
            task_type=None,
            inference_mode=False,
            **build_lora_config_kwargs(
                raw,
                use_duration_predictor=bool(model.cfg.use_duration_predictor),
            ),
        ),
    )
    return peft_model


def is_lora_adapter_dir(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.is_dir():
        return False
    if not (candidate / LORA_ADAPTER_CONFIG_NAME).is_file():
        return False
    return any((candidate / name).is_file() for name in LORA_ADAPTER_STATE_NAMES)


def _character_lora_error(config_path: Path, detail: str) -> ValueError:
    return ValueError(
        "Unsupported character LoRA adapter configuration for the B1 inference runtime: "
        f"{detail} ({config_path})."
    )


def _is_none_or_empty(value: Any) -> bool:
    return value is None or value == [] or value == {}


def _validated_character_lora_adapter_config(path: str | Path) -> Any:
    config_path = Path(path) / LORA_ADAPTER_CONFIG_NAME
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LoRA adapter config JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"LoRA adapter config must contain a JSON object: {config_path}")

    unknown_fields = sorted(set(payload) - _KNOWN_CHARACTER_LORA_CONFIG_FIELDS)
    if unknown_fields:
        raise _character_lora_error(
            config_path,
            f"unknown PEFT fields are not accepted: {unknown_fields}",
        )
    if payload.get("peft_type", "LORA") != "LORA":
        raise _character_lora_error(config_path, "peft_type must be 'LORA'")
    if payload.get("init_lora_weights", True) is not True:
        raise _character_lora_error(config_path, "init_lora_weights must be exactly true")
    if payload.get("bias", "none") != "none":
        raise _character_lora_error(config_path, "bias must be exactly 'none'")
    if payload.get("lora_bias", False) is not False:
        raise _character_lora_error(config_path, "lora_bias must be false")

    rank_pattern = payload.get("rank_pattern", {})
    alpha_pattern = payload.get("alpha_pattern", {})
    if not isinstance(rank_pattern, dict) or rank_pattern:
        raise _character_lora_error(config_path, "rank_pattern must be an empty object")
    if not isinstance(alpha_pattern, dict) or alpha_pattern:
        raise _character_lora_error(config_path, "alpha_pattern must be an empty object")

    rank = payload.get("r")
    alpha = payload.get("lora_alpha")
    dropout = payload.get("lora_dropout", CHARACTER_LORA_DROPOUT)
    if not isinstance(rank, int) or isinstance(rank, bool) or rank != CHARACTER_LORA_R:
        raise _character_lora_error(config_path, f"r must be {CHARACTER_LORA_R}")
    if not isinstance(alpha, int) or isinstance(alpha, bool) or alpha != CHARACTER_LORA_ALPHA:
        raise _character_lora_error(
            config_path,
            f"lora_alpha must be {CHARACTER_LORA_ALPHA}",
        )
    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or float(dropout) != CHARACTER_LORA_DROPOUT
    ):
        raise _character_lora_error(
            config_path,
            f"lora_dropout must be {CHARACTER_LORA_DROPOUT}",
        )

    target_modules = payload.get("target_modules")
    target_modules_valid = (
        isinstance(target_modules, str) and bool(target_modules.strip())
    ) or (
        isinstance(target_modules, list)
        and bool(target_modules)
        and all(isinstance(item, str) and bool(item.strip()) for item in target_modules)
    )
    if not target_modules_valid:
        raise _character_lora_error(
            config_path,
            "target_modules must identify at least one supported nn.Linear",
        )

    modules_to_save = payload.get("modules_to_save")
    if modules_to_save is None:
        normalized_modules_to_save: tuple[str, ...] = ()
    elif isinstance(modules_to_save, list) and all(
        isinstance(item, str) for item in modules_to_save
    ):
        normalized_modules_to_save = tuple(modules_to_save)
    else:
        raise _character_lora_error(
            config_path,
            "modules_to_save must be null or ['duration_predictor']",
        )
    if normalized_modules_to_save not in ((), CHARACTER_LORA_MODULES_TO_SAVE):
        raise _character_lora_error(
            config_path,
            "modules_to_save must be null or exactly ['duration_predictor']",
        )

    for field in (
        "ensure_weight_tying",
        "fan_in_fan_out",
        "use_dora",
        "use_qalora",
        "use_rslora",
    ):
        if payload.get(field, False) is not False:
            raise _character_lora_error(config_path, f"{field} is not supported")

    for field in (
        "alora_invocation_tokens",
        "arrow_config",
        "corda_config",
        "eva_config",
        "exclude_modules",
        "layer_replication",
        "layers_pattern",
        "layers_to_transform",
        "loftq_config",
        "megatron_config",
        "target_parameters",
        "trainable_token_indices",
    ):
        if not _is_none_or_empty(payload.get(field)):
            raise _character_lora_error(config_path, f"{field} is not supported")
    if payload.get("task_type") is not None:
        raise _character_lora_error(config_path, "task_type must be null")

    lora_config_cls, _, _ = _require_peft()
    peft_config = lora_config_cls.from_peft_type(**payload)
    peft_config.inference_mode = True
    return peft_config


def lora_adapter_mutates_base_parameters(path: str | Path) -> bool:
    """Validate the character-runtime config boundary; supported adapters never mutate base."""
    _validated_character_lora_adapter_config(path)
    return False


def _unwrapped_model(model: torch.nn.Module) -> torch.nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        return get_base_model()
    return model


def _peft_target_module_names(model: torch.nn.Module, peft_config: Any) -> set[str]:
    from peft.tuners.tuners_utils import (
        _ExcludedModule,
        _maybe_include_all_linear_layers,
        check_target_module_exists,
    )

    base_model = _unwrapped_model(model)
    peft_config = _maybe_include_all_linear_layers(peft_config, base_model)
    targets: set[str] = set()
    for name, _ in base_model.named_modules():
        if not name:
            continue
        match = check_target_module_exists(peft_config, name)
        if match and not isinstance(match, _ExcludedModule):
            targets.add(name)
    return targets


def _validate_character_target_modules(
    model: torch.nn.Module,
    peft_config: Any,
    *,
    config_path: Path,
) -> set[str]:
    _, peft_model_cls, _ = _require_peft()
    if isinstance(model, peft_model_cls) or getattr(model, "peft_config", None):
        raise _character_lora_error(
            config_path,
            "the runtime model already contains PEFT adapter state",
        )

    base_model = _unwrapped_model(model)
    target_names = _peft_target_module_names(base_model, peft_config)
    if not target_names:
        raise _character_lora_error(config_path, "target_modules matched no model module")

    unsupported = sorted(
        name
        for name in target_names
        if not isinstance(base_model.get_submodule(name), torch.nn.Linear)
    )
    if unsupported:
        raise _character_lora_error(
            config_path,
            f"only ordinary nn.Linear targets are supported, got {unsupported[:3]}",
        )

    if peft_config.modules_to_save:
        try:
            base_model.get_submodule("duration_predictor")
        except AttributeError as exc:
            raise _character_lora_error(
                config_path,
                "modules_to_save requests duration_predictor but the model has none",
            ) from exc
        conflicts = sorted(
            name
            for name in target_names
            if name == "duration_predictor" or name.startswith("duration_predictor.")
        )
        if conflicts:
            raise _character_lora_error(
                config_path,
                "duration_predictor cannot be both a LoRA target and modules_to_save",
            )
    return target_names


def _expected_lora_destination_shapes(
    model: torch.nn.Module,
    *,
    adapter_name: str,
    target_names: set[str],
    rank: int,
) -> dict[str, torch.Size]:
    base_model = _unwrapped_model(model)
    destinations: dict[str, torch.Size] = {}
    for name in target_names:
        module = base_model.get_submodule(name)
        assert isinstance(module, torch.nn.Linear)
        prefix = f"base_model.model.{name}"
        destinations[f"{prefix}.lora_A.{adapter_name}.weight"] = torch.Size(
            (rank, module.in_features)
        )
        destinations[f"{prefix}.lora_B.{adapter_name}.weight"] = torch.Size(
            (module.out_features, rank)
        )
    return destinations


def _expected_auxiliary_destinations(
    model: torch.nn.Module,
    peft_config: Any,
    *,
    adapter_name: str,
) -> dict[str, tuple[str, torch.Size]]:
    from peft.utils.other import ModulesToSaveWrapper

    if not peft_config.modules_to_save:
        return {}

    base_model = _unwrapped_model(model)
    module = base_model.get_submodule("duration_predictor")
    proxy = SimpleNamespace(
        _adapters={adapter_name},
        modules_to_save={adapter_name: module},
    )
    key_map = ModulesToSaveWrapper.adapter_state_dict_load_map(proxy, adapter_name)
    module_state = module.state_dict()
    prefix = "base_model.model.duration_predictor"
    destinations: dict[str, tuple[str, torch.Size]] = {}
    for source_suffix, destination_suffix in key_map.items():
        source_tensor = module_state.get(source_suffix)
        if source_tensor is None:
            raise RuntimeError(
                "PEFT modules_to_save mapping referenced missing duration_predictor state "
                f"{source_suffix!r}."
            )
        destinations[f"{prefix}.{source_suffix}"] = (
            f"{prefix}.{destination_suffix}",
            source_tensor.shape,
        )
    return destinations


def _peft_lora_payload_destination(key: str, *, adapter_name: str) -> str:
    from peft.utils.save_and_load import _insert_adapter_name_into_state_dict

    transformed = _insert_adapter_name_into_state_dict(
        {key: None},
        adapter_name=adapter_name,
        parameter_prefix="lora_",
    )
    return next(iter(transformed))


def _record_expected_state(
    expected_state: dict[str, torch.Tensor],
    *,
    destination: str,
    tensor: torch.Tensor,
    path: str | Path,
) -> None:
    if destination in expected_state:
        raise ValueError(
            "Unsupported character LoRA adapter state: multiple payload entries map to "
            f"destination {destination!r} in {path}."
        )
    expected_state[destination] = tensor.detach().clone()


def _validate_character_lora_adapter_state(
    model: torch.nn.Module,
    path: str | Path,
    *,
    peft_config: Any,
    adapter_name: str,
    target_names: set[str],
) -> tuple[tuple[tuple[str, torch.Tensor], ...], tuple[tuple[str, torch.Tensor], ...]]:
    from peft.utils.save_and_load import load_peft_weights

    state = load_peft_weights(str(path), device="cpu")
    if not isinstance(state, Mapping):
        raise ValueError(f"LoRA adapter state must contain a tensor mapping: {path}")

    lora_shapes = _expected_lora_destination_shapes(
        model,
        adapter_name=adapter_name,
        target_names=target_names,
        rank=int(peft_config.r),
    )
    auxiliary = _expected_auxiliary_destinations(
        model,
        peft_config,
        adapter_name=adapter_name,
    )
    required_shapes = dict(lora_shapes)
    required_shapes.update(destination for destination in auxiliary.values())

    unsafe_keys: list[str] = []
    expected_state: dict[str, torch.Tensor] = {}
    validated_payload: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            unsafe_keys.append(repr(key))
            continue
        if tensor.layout != torch.strided:
            raise ValueError(
                "Unsupported character LoRA adapter state: tensor layout for "
                f"{key!r} must be torch.strided, got {tensor.layout} ({path})."
            )
        if not tensor.is_floating_point():
            unsafe_keys.append(key)
            continue

        auxiliary_destination = auxiliary.get(key)
        if auxiliary_destination is not None:
            destination, expected_shape = auxiliary_destination
        elif any(
            marker in key
            for marker in (".original_module.", ".modules_to_save.", ".token_adapter.")
        ):
            unsafe_keys.append(key)
            continue
        elif "lora_" in key:
            destination = _peft_lora_payload_destination(key, adapter_name=adapter_name)
            expected_shape = lora_shapes.get(destination)
            if expected_shape is None:
                unsafe_keys.append(key)
                continue
        else:
            unsafe_keys.append(key)
            continue

        if tensor.shape != expected_shape:
            raise ValueError(
                "Unsupported character LoRA adapter state: shape mismatch for "
                f"{key!r}; expected {tuple(expected_shape)}, got {tuple(tensor.shape)} ({path})."
            )
        _record_expected_state(
            expected_state,
            destination=destination,
            tensor=tensor,
            path=path,
        )
        validated_payload[key] = tensor.detach().clone()

    if unsafe_keys:
        sample = ", ".join(repr(key) for key in unsafe_keys[:3])
        if len(unsafe_keys) > 3:
            sample += f", ... ({len(unsafe_keys)} total)"
        raise ValueError(
            "Unsupported character LoRA adapter state: payload contains unrecognized, "
            f"shared-base, or invalid adapter entries in {path}: {sample}."
        )

    missing = sorted(set(required_shapes) - set(expected_state))
    if missing:
        sample = ", ".join(repr(key) for key in missing[:3])
        if len(missing) > 3:
            sample += f", ... ({len(missing)} total)"
        raise ValueError(
            "Incomplete character LoRA adapter state: missing required A/B or "
            f"modules_to_save tensors in {path}: {sample}."
        )
    return tuple(validated_payload.items()), tuple(expected_state.items())


def apply_preflighted_lora_adapter(
    model: torch.nn.Module,
    preflight: LoraAdapterPreflight,
) -> torch.nn.Module:
    """Inject and load the exact config and payload already accepted by preflight."""
    _, peft_model_cls, get_peft_model = _require_peft()
    if isinstance(model, peft_model_cls) or getattr(model, "peft_config", None):
        raise RuntimeError("Character runtime model already contains a PEFT adapter.")

    peft_model = get_peft_model(
        model,
        preflight.peft_config,
        adapter_name=preflight.adapter_name,
    )
    from peft.utils.save_and_load import set_peft_model_state_dict

    set_peft_model_state_dict(
        peft_model,
        dict(preflight.adapter_state),
        adapter_name=preflight.adapter_name,
    )
    peft_model.set_adapter(preflight.adapter_name)
    return peft_model


def validate_lora_adapter_applied_state(
    model: torch.nn.Module,
    preflight: LoraAdapterPreflight,
) -> None:
    """Verify complete values and immutable single-adapter inference state."""
    _, peft_model_cls, _ = _require_peft()
    failures: list[str] = []
    if not isinstance(model, peft_model_cls):
        failures.append("model is not a PEFT model")

    configs = getattr(model, "peft_config", {})
    if set(configs) != {preflight.adapter_name}:
        failures.append(f"loaded adapter set is {sorted(configs)!r}")
    else:
        config = configs[preflight.adapter_name]
        if not bool(getattr(config, "inference_mode", False)):
            failures.append("adapter is not configured for inference")

    active = getattr(model, "active_adapters", [])
    if callable(active):
        active = active()
    if isinstance(active, str):
        active = [active]
    if list(active) != [preflight.adapter_name]:
        failures.append(f"active adapters are {list(active)!r}")

    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils.other import ModulesToSaveWrapper

    for module in model.modules():
        if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
            if bool(getattr(module, "merged", False)):
                failures.append("an adapter layer is merged")
            if bool(getattr(module, "disable_adapters", False)):
                failures.append("an adapter layer is disabled")
    if model.training:
        failures.append("model is not in eval mode")

    actual_state = model.state_dict()
    for destination, expected in preflight.expected_applied_state:
        actual = actual_state.get(destination)
        if actual is None:
            failures.append(f"missing {destination!r}")
            continue
        if actual.shape != expected.shape:
            failures.append(
                f"shape mismatch for {destination!r}: expected {tuple(expected.shape)}, "
                f"got {tuple(actual.shape)}"
            )
            continue
        converted = expected.to(device=actual.device, dtype=actual.dtype)
        if not torch.equal(actual.detach(), converted):
            failures.append(f"value mismatch for {destination!r}")

    if failures:
        sample = "; ".join(failures[:3])
        if len(failures) > 3:
            sample += f"; ... ({len(failures)} total)"
        raise RuntimeError(f"Character LoRA post-load verification failed: {sample}.")


def preflight_lora_adapter(
    model: torch.nn.Module,
    path: str | Path,
    *,
    adapter_name: str = "character",
) -> LoraAdapterPreflight:
    """Validate the complete supported first/only character adapter without mutation."""
    config_path = Path(path) / LORA_ADAPTER_CONFIG_NAME
    peft_config = _validated_character_lora_adapter_config(path)
    target_names = _validate_character_target_modules(
        model,
        peft_config,
        config_path=config_path,
    )
    adapter_state, expected_applied_state = _validate_character_lora_adapter_state(
        model,
        path,
        peft_config=peft_config,
        adapter_name=adapter_name,
        target_names=target_names,
    )
    return LoraAdapterPreflight(
        adapter_name=adapter_name,
        peft_config=peft_config,
        adapter_state=adapter_state,
        expected_applied_state=expected_applied_state,
    )


def load_lora_adapter(
    model: TextToLatentRFDiT,
    adapter_path: str | Path,
    *,
    is_trainable: bool,
    adapter_name: str = "default",
    torch_device: str | None = None,
) -> torch.nn.Module:
    """Broader PEFT loader retained for training, resume, and checkpoint conversion."""
    _, peft_model_cls, _ = _require_peft()
    if isinstance(model, peft_model_cls):
        if adapter_name not in model.peft_config:
            model.load_adapter(
                str(adapter_path),
                adapter_name=adapter_name,
                is_trainable=is_trainable,
                torch_device=torch_device,
            )
        model.set_adapter(adapter_name)
        return model
    return peft_model_cls.from_pretrained(
        model,
        str(adapter_path),
        adapter_name=adapter_name,
        is_trainable=is_trainable,
        torch_device=torch_device,
    )


def model_supports_lora_adapters(model: torch.nn.Module) -> bool:
    _, peft_model_cls, _ = _require_peft()
    return isinstance(model, peft_model_cls)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    total = sum(int(param.numel()) for param in model.parameters())
    return trainable, total
