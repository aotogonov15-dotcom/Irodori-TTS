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

_SAFE_DYNAMIC_LORA_INITIALIZATIONS = {
    "gaussian",
    "eva",
    "orthogonal",
}
_SUPPORTED_LORA_BIASES = {
    "none",
    "all",
    "lora_only",
}
_BASE_MUTATING_LORA_INITIALIZATIONS = {
    "olora",
    "corda",
    "loftq",
}


@dataclass(frozen=True)
class LoraAdapterPreflight:
    """Dynamic-loading facts validated before PEFT can mutate the model."""

    bias: str

    @property
    def mutates_base_parameters(self) -> bool:
        return self.bias != "none"

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


def _validated_lora_adapter_config(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    config_path = Path(path) / LORA_ADAPTER_CONFIG_NAME
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LoRA adapter config JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"LoRA adapter config must contain a JSON object: {config_path}")

    raw_initialization = payload.get("init_lora_weights", True)
    initialization_is_safe = isinstance(raw_initialization, bool) or (
        isinstance(raw_initialization, str)
        and raw_initialization in _SAFE_DYNAMIC_LORA_INITIALIZATIONS
    )

    mutates_base = isinstance(raw_initialization, str) and (
        raw_initialization == "pissa"
        or raw_initialization.startswith("pissa_niter_")
        or raw_initialization in _BASE_MUTATING_LORA_INITIALIZATIONS
    )
    if mutates_base:
        raise ValueError(
            "Unsupported LoRA adapter configuration for dynamic runtime LoRA: "
            f"init_lora_weights={raw_initialization!r} in {config_path} can persistently "
            "modify shared base parameters, so disabling the adapter cannot guarantee "
            "restoration of the effective base model."
        )
    if not initialization_is_safe:
        raise ValueError(
            "Unsupported LoRA adapter configuration for dynamic runtime LoRA: "
            f"init_lora_weights={raw_initialization!r} in {config_path} is not a "
            "recognized non-mutating initialization, so safe base restoration cannot "
            "be proven."
        )

    bias = payload.get("bias", "none")
    if not isinstance(bias, str) or bias not in _SUPPORTED_LORA_BIASES:
        raise ValueError(
            f"Unsupported LoRA adapter bias={bias!r} in {config_path}. "
            "Expected an exact canonical value from: none, all, lora_only."
        )
    return config_path, payload, bias


def lora_adapter_mutates_base_parameters(path: str | Path) -> bool:
    """Validate config-only safety and report supported persistent bias mutation."""
    _, _, bias = _validated_lora_adapter_config(path)
    return bias != "none"


def _unwrapped_model(model: torch.nn.Module) -> torch.nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        return get_base_model()
    return model


def _canonical_base_parameter_name(name: str) -> str:
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    while ".base_layer." in name:
        name = name.replace(".base_layer.", ".")
    while ".original_module." in name:
        name = name.replace(".original_module.", ".")
    return name


def _peft_target_module_names(model: torch.nn.Module, peft_config: Any) -> set[str]:
    # These are the same matching helpers and existing-adapter exclusion used by PEFT 0.18.1
    # while injecting a LoRA adapter. Keeping this dry-run tied to PEFT avoids subtly different
    # rank_pattern behavior.
    from peft.tuners.tuners_utils import (
        BaseTunerLayer,
        _ExcludedModule,
        _maybe_include_all_linear_layers,
        check_target_module_exists,
    )

    base_model = _unwrapped_model(model)
    peft_config = _maybe_include_all_linear_layers(peft_config, base_model)
    named_modules = list(base_model.named_modules())
    existing_adapter_prefixes = [
        f"{name}." for name, module in named_modules if isinstance(module, BaseTunerLayer)
    ]
    targets: set[str] = set()
    for name, _ in named_modules:
        if not name or any(name.startswith(prefix) for prefix in existing_adapter_prefixes):
            continue
        match = check_target_module_exists(peft_config, name)
        if match and not isinstance(match, _ExcludedModule):
            targets.add(name)

    target_parameters = getattr(peft_config, "target_parameters", None) or []
    if target_parameters:
        parameter_names = {
            _canonical_base_parameter_name(name)
            for name, _ in base_model.named_parameters()
            if ".lora_" not in name and ".modules_to_save." not in name
        }
        for name in parameter_names:
            if name in target_parameters or any(
                name.endswith(f".{target}") for target in target_parameters
            ):
                targets.add(name)
    return targets


def _validate_orthogonal_ranks(
    model: torch.nn.Module,
    peft_config: Any,
    *,
    config_path: Path,
) -> set[str]:
    from peft.utils.other import get_pattern_key

    target_names = _peft_target_module_names(model, peft_config)
    rank_pattern = peft_config.rank_pattern
    for target_name in sorted(target_names):
        rank_key = get_pattern_key(rank_pattern.keys(), target_name)
        rank = rank_pattern.get(rank_key, peft_config.r)
        if rank % 2:
            raise ValueError(
                "Unsupported LoRA adapter configuration for dynamic runtime LoRA: "
                f"orthogonal initialization requires an even effective rank, but {target_name!r} "
                f"receives r={rank} in {config_path}."
            )
    return target_names


def _peft_auxiliary_payload_destinations(
    model: torch.nn.Module,
    peft_config: Any,
    *,
    adapter_name: str,
) -> dict[str, str]:
    """Map exact saved auxiliary keys to the state keys PEFT 0.18.1 will write."""
    from peft.utils.other import ModulesToSaveWrapper, TrainableTokensWrapper

    base_model = _unwrapped_model(model)
    named_modules = list(base_model.named_modules(remove_duplicate=False))
    destinations: dict[str, str] = {}

    configured_modules = peft_config.modules_to_save or []
    for name, module in named_modules:
        if not name or not any(name.endswith(target) for target in configured_modules):
            continue
        if isinstance(module, ModulesToSaveWrapper):
            storage_module = module.original_module
        else:
            storage_module = module

        # Use PEFT's wrapper-owned source-to-destination map without mutating the live
        # model. The proxy supplies precisely the attributes used by 0.18.1's helper.
        proxy = SimpleNamespace(
            _adapters={adapter_name},
            modules_to_save={adapter_name: storage_module},
        )
        key_map = ModulesToSaveWrapper.adapter_state_dict_load_map(proxy, adapter_name)
        module_name = f"base_model.model.{name}"
        for source_suffix, destination_suffix in key_map.items():
            destinations[f"{module_name}.{source_suffix}"] = (
                f"{module_name}.{destination_suffix}"
            )

    trainable_token_indices = getattr(peft_config, "trainable_token_indices", None)
    if trainable_token_indices is not None:
        if not isinstance(trainable_token_indices, dict):
            from peft.utils.other import _get_input_embeddings_name

            input_name = _get_input_embeddings_name(base_model, "embed_tokens")
            trainable_token_indices = {input_name: trainable_token_indices}

        token_proxy = SimpleNamespace(token_adapter=SimpleNamespace(tied_adapter=None))
        key_map = TrainableTokensWrapper.adapter_state_dict_load_map(
            token_proxy,
            adapter_name,
        )
        for target_name in trainable_token_indices:
            matching_names = [
                name
                for name, _ in named_modules
                if name and name.endswith(target_name)
            ]
            for name in matching_names:
                module_name = f"base_model.model.{name}"
                for source_suffix, destination_suffix in key_map.items():
                    destinations[f"{module_name}.{source_suffix}"] = (
                        f"{module_name}.{destination_suffix}"
                    )

    return destinations


def _shared_base_bias_names(model: torch.nn.Module) -> set[str]:
    return {
        _canonical_base_parameter_name(name)
        for name, _ in _unwrapped_model(model).named_parameters()
        if name.endswith(".bias")
        and ".lora_" not in name
        and ".modules_to_save." not in name
    }


def _validate_lora_adapter_state(
    model: torch.nn.Module,
    path: str | Path,
    *,
    bias: str,
    target_names: set[str],
    peft_config: Any,
    adapter_name: str,
) -> None:
    # This is PEFT's own local-file selection and deserialization path (safetensors first,
    # then adapter_model.bin with weights_only=True), so the keys checked here are precisely
    # the payload that set_peft_model_state_dict will receive.
    from peft.utils.save_and_load import load_peft_weights

    state = load_peft_weights(str(path), device="cpu")
    if not isinstance(state, Mapping):
        raise ValueError(f"LoRA adapter state must contain a tensor mapping: {path}")

    auxiliary_destinations = _peft_auxiliary_payload_destinations(
        model,
        peft_config,
        adapter_name=adapter_name,
    )
    shared_biases = _shared_base_bias_names(model)
    unsafe_keys: list[str] = []
    for key in state:
        if not isinstance(key, str):
            unsafe_keys.append(repr(key))
            continue

        # These are exact source keys returned by PEFT's auxiliary wrapper load maps.
        # In particular, original_module and pre-existing adapter namespaces cannot
        # pass merely because they are located beneath the same logical module.
        if key in auxiliary_destinations:
            continue

        # Wrapper internals are never valid serialized source paths unless PEFT's
        # exact load map above says otherwise. Check them before the LoRA prefix,
        # since an attacker-controlled adapter namespace can itself contain "lora_".
        if any(
            marker in key
            for marker in (".original_module.", ".modules_to_save.", ".token_adapter.")
        ):
            unsafe_keys.append(key)
            continue

        # PEFT inserts the runtime adapter name into every LoRA-prefixed state key.
        if "lora_" in key:
            continue

        canonical_key = _canonical_base_parameter_name(key)
        if canonical_key.endswith(".bias") and canonical_key in shared_biases:
            if bias == "all":
                continue
            if bias == "lora_only" and canonical_key.removesuffix(".bias") in target_names:
                continue

        # Every remaining key is passed through unchanged to model.load_state_dict(strict=False).
        # If it matches, it writes shared state; if it does not, rejecting it is the fail-closed
        # behavior required for an unrecognized dynamic adapter payload.
        unsafe_keys.append(key)

    if unsafe_keys:
        sample = ", ".join(repr(key) for key in unsafe_keys[:3])
        if len(unsafe_keys) > 3:
            sample += f", ... ({len(unsafe_keys)} total)"
        raise ValueError(
            "Unsupported LoRA adapter state for dynamic runtime LoRA: payload contains "
            f"unrecognized shared-base writes in {path}: {sample}."
        )


def preflight_lora_adapter(
    model: torch.nn.Module,
    path: str | Path,
    *,
    adapter_name: str = "default",
) -> LoraAdapterPreflight:
    """Validate config, load payload, ranks, and loaded-adapter conflicts without mutation."""
    config_path, payload, bias = _validated_lora_adapter_config(path)
    lora_config_cls, _, _ = _require_peft()
    peft_config = lora_config_cls.from_pretrained(str(path))

    target_names = _peft_target_module_names(model, peft_config)
    if payload.get("init_lora_weights", True) == "orthogonal":
        target_names = _validate_orthogonal_ranks(
            model,
            peft_config,
            config_path=config_path,
        )

    loaded_configs = getattr(model, "peft_config", {})
    if bias != "none" and any(
        getattr(config, "bias", "none") != "none" for config in loaded_configs.values()
    ):
        raise ValueError(
            "Unsupported LoRA adapter combination for dynamic runtime LoRA: PEFT supports "
            "only one loaded adapter with bias != 'none'."
        )

    _validate_lora_adapter_state(
        model,
        path,
        bias=bias,
        target_names=target_names,
        peft_config=peft_config,
        adapter_name=adapter_name,
    )
    return LoraAdapterPreflight(bias=bias)


def load_lora_adapter(
    model: TextToLatentRFDiT,
    adapter_path: str | Path,
    *,
    is_trainable: bool,
    adapter_name: str = "default",
    torch_device: str | None = None,
) -> torch.nn.Module:
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
