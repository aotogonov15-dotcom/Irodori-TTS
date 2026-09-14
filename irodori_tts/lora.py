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
    expected_applied_state: tuple[tuple[str, torch.Tensor], ...]

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

    return targets


def _peft_target_parameter_names(model: torch.nn.Module, peft_config: Any) -> set[str]:
    target_parameters = getattr(peft_config, "target_parameters", None) or []
    if not target_parameters:
        return set()

    return {
        canonical_name
        for name, _ in _unwrapped_model(model).named_parameters()
        if ".lora_" not in name and ".modules_to_save." not in name
        for canonical_name in (_canonical_base_parameter_name(name),)
        if canonical_name in target_parameters
        or any(canonical_name.endswith(f".{target}") for target in target_parameters)
    }


def _validate_wrapper_topology(
    model: torch.nn.Module,
    peft_config: Any,
    *,
    loaded_configs: Mapping[str, Any],
) -> set[str]:
    """Reject PEFT 0.18.1 wrapper combinations known to mutate before failing."""
    from peft.tuners.tuners_utils import (
        BaseTunerLayer,
        _ExcludedModule,
        check_target_module_exists,
    )
    from peft.utils.other import ModulesToSaveWrapper

    base_model = _unwrapped_model(model)
    named_modules = list(base_model.named_modules(remove_duplicate=False))

    target_parameters = getattr(peft_config, "target_parameters", None) or []
    if target_parameters and any(
        getattr(config, "target_parameters", None) for config in loaded_configs.values()
    ):
        raise ValueError(
            "Unsupported LoRA adapter combination for dynamic runtime LoRA: PEFT 0.18.1 "
            "supports only one loaded adapter with target_parameters."
        )

    # A ModulesToSaveWrapper hides its original descendants below `.original_module`.
    # PEFT's target matcher therefore misses a later adapter targeting the wrapper or
    # one of those logical descendants and can report success without adding LoRA state.
    for wrapper_name, module in named_modules:
        if not wrapper_name or not isinstance(module, ModulesToSaveWrapper):
            continue
        logical_modules = (
            wrapper_name if not suffix else f"{wrapper_name}.{suffix}"
            for suffix, _ in module.original_module.named_modules()
        )
        for logical_name in logical_modules:
            match = check_target_module_exists(peft_config, logical_name)
            if match and not isinstance(match, _ExcludedModule):
                raise ValueError(
                    "Unsupported LoRA adapter topology for dynamic runtime LoRA: target module "
                    f"{logical_name!r} is inside existing ModulesToSaveWrapper {wrapper_name!r}."
                )

        for suffix, _ in module.original_module.named_parameters():
            logical_name = f"{wrapper_name}.{suffix}"
            if logical_name in target_parameters or any(
                logical_name.endswith(f".{target}") for target in target_parameters
            ):
                raise ValueError(
                    "Unsupported LoRA adapter topology for dynamic runtime LoRA: target parameter "
                    f"{logical_name!r} is inside existing ModulesToSaveWrapper {wrapper_name!r}."
                )

    # Wrapping a module that already contains LoRA storage copies/nests another
    # adapter's topology. B1 does not need that composition, so reject it cleanly.
    configured_modules = peft_config.modules_to_save or []
    modules_to_save_names = {
        name
        for name, _ in named_modules
        if name and any(name.endswith(target) for target in configured_modules)
    }
    for saved_name in modules_to_save_names:
        for existing_name, module in named_modules:
            if not isinstance(module, BaseTunerLayer):
                continue
            if existing_name == saved_name or existing_name.startswith(f"{saved_name}."):
                raise ValueError(
                    "Unsupported LoRA adapter topology for dynamic runtime LoRA: "
                    f"modules_to_save target {saved_name!r} contains existing LoRA layer "
                    f"{existing_name!r}."
                )

    matched_parameters = _peft_target_parameter_names(model, peft_config)
    for parameter_name in sorted(matched_parameters):
        module_name, _, _ = parameter_name.rpartition(".")
        try:
            module = base_model.get_submodule(module_name)
        except AttributeError:
            continue
        if isinstance(module, BaseTunerLayer) and module.__class__.__name__ != "ParamWrapper":
            raise ValueError(
                "Unsupported LoRA adapter topology for dynamic runtime LoRA: target_parameters "
                f"entry {parameter_name!r} belongs to existing LoRA wrapper {module_name!r}."
            )
    return matched_parameters


def _has_trainable_tokens_wrapper(model: torch.nn.Module) -> bool:
    from peft.utils.other import TrainableTokensWrapper

    return any(
        isinstance(module, TrainableTokensWrapper)
        for module in _unwrapped_model(model).modules()
    )


def _validate_orthogonal_ranks(
    model: torch.nn.Module,
    peft_config: Any,
    *,
    config_path: Path,
) -> set[str]:
    from peft.utils.other import get_pattern_key

    target_names = _peft_target_module_names(model, peft_config)
    rank_targets = target_names | _peft_target_parameter_names(model, peft_config)
    rank_pattern = peft_config.rank_pattern
    for target_name in sorted(rank_targets):
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


def _expected_new_lora_destinations(
    model: torch.nn.Module,
    peft_config: Any,
    *,
    adapter_name: str,
    target_module_names: set[str],
    target_parameter_names: set[str],
) -> set[str]:
    from peft.tuners.tuners_utils import BaseTunerLayer

    base_model = _unwrapped_model(model)
    target_modules: dict[str, torch.nn.Module] = {}
    for name in target_module_names:
        target_modules[name] = base_model.get_submodule(name)
    for parameter_name in target_parameter_names:
        module_name, _, _ = parameter_name.rpartition(".")
        target_modules[module_name] = base_model.get_submodule(module_name)

    destinations: set[str] = set()
    for name, module in target_modules.items():
        if isinstance(module, BaseTunerLayer):
            module = module.get_base_layer()
        prefix = f"base_model.model.{name}"
        if isinstance(module, torch.nn.Embedding):
            destinations.update(
                {
                    f"{prefix}.lora_embedding_A.{adapter_name}",
                    f"{prefix}.lora_embedding_B.{adapter_name}",
                }
            )
        else:
            destinations.update(
                {
                    f"{prefix}.lora_A.{adapter_name}.weight",
                    f"{prefix}.lora_B.{adapter_name}.weight",
                }
            )
            if getattr(peft_config, "lora_bias", False):
                destinations.update(
                    {
                        f"{prefix}.lora_A.{adapter_name}.bias",
                        f"{prefix}.lora_B.{adapter_name}.bias",
                    }
                )
        if getattr(peft_config, "use_dora", False):
            destinations.add(f"{prefix}.lora_magnitude_vector.{adapter_name}.weight")
    return destinations


def _peft_lora_payload_destination(key: str, *, adapter_name: str) -> str:
    # Use PEFT 0.18.1's own adapter-name insertion instead of approximating its
    # rpartition/replacement behavior. Mirror its legacy DoRA key normalization too.
    from peft.utils.save_and_load import _insert_adapter_name_into_state_dict

    transformed = _insert_adapter_name_into_state_dict(
        {key: None},
        adapter_name=adapter_name,
        parameter_prefix="lora_",
    )
    destination = next(iter(transformed))
    old_dora_suffix = f"lora_magnitude_vector.{adapter_name}"
    if destination.endswith(old_dora_suffix):
        destination += ".weight"
    return destination


def _record_expected_state(
    expected_state: dict[str, torch.Tensor],
    *,
    destination: str,
    tensor: torch.Tensor,
    path: str | Path,
) -> None:
    if destination in expected_state:
        raise ValueError(
            "Unsupported LoRA adapter state for dynamic runtime LoRA: multiple payload "
            f"entries map to destination {destination!r} in {path}."
        )
    expected_state[destination] = tensor.detach().clone()


def _validate_lora_adapter_state(
    model: torch.nn.Module,
    path: str | Path,
    *,
    bias: str,
    target_names: set[str],
    target_parameter_names: set[str],
    peft_config: Any,
    adapter_name: str,
) -> tuple[tuple[str, torch.Tensor], ...]:
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
    lora_destinations = _expected_new_lora_destinations(
        model,
        peft_config,
        adapter_name=adapter_name,
        target_module_names=target_names,
        target_parameter_names=target_parameter_names,
    )
    shared_biases = _shared_base_bias_names(model)
    unsafe_keys: list[str] = []
    expected_state: dict[str, torch.Tensor] = {}
    new_adapter_destinations = 0
    for key, tensor in state.items():
        if not isinstance(key, str):
            unsafe_keys.append(repr(key))
            continue
        if not isinstance(tensor, torch.Tensor):
            unsafe_keys.append(key)
            continue

        # These are exact source keys returned by PEFT's auxiliary wrapper load maps.
        # In particular, original_module and pre-existing adapter namespaces cannot
        # pass merely because they are located beneath the same logical module.
        if key in auxiliary_destinations:
            _record_expected_state(
                expected_state,
                destination=auxiliary_destinations[key],
                tensor=tensor,
                path=path,
            )
            new_adapter_destinations += 1
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
            destination = _peft_lora_payload_destination(key, adapter_name=adapter_name)
            if destination not in lora_destinations:
                unsafe_keys.append(key)
                continue
            _record_expected_state(
                expected_state,
                destination=destination,
                tensor=tensor,
                path=path,
            )
            new_adapter_destinations += 1
            continue

        canonical_key = _canonical_base_parameter_name(key)
        if canonical_key.endswith(".bias") and canonical_key in shared_biases:
            if bias == "all":
                _record_expected_state(
                    expected_state,
                    destination=key,
                    tensor=tensor,
                    path=path,
                )
                continue
            if bias == "lora_only" and canonical_key.removesuffix(".bias") in target_names:
                _record_expected_state(
                    expected_state,
                    destination=key,
                    tensor=tensor,
                    path=path,
                )
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
            f"unrecognized shared-base writes or non-applicable adapter entries in {path}: "
            f"{sample}."
        )
    if new_adapter_destinations == 0:
        raise ValueError(
            "Unsupported LoRA adapter state for dynamic runtime LoRA: no adapter-owned "
            f"tensor in {path} has an actual destination for the new adapter."
        )
    return tuple(expected_state.items())


def validate_lora_adapter_applied_state(
    model: torch.nn.Module,
    preflight: LoraAdapterPreflight,
) -> None:
    """Prove every accepted payload tensor reached its exact post-load destination."""
    actual_state = model.state_dict()
    failures: list[str] = []
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
        raise RuntimeError(
            "Dynamic LoRA adapter load did not apply every preflight-accepted tensor: "
            f"{sample}."
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
    target_parameter_names = _validate_wrapper_topology(
        model,
        peft_config,
        loaded_configs=loaded_configs,
    )
    if bias != "none" and any(
        getattr(config, "bias", "none") != "none" for config in loaded_configs.values()
    ):
        raise ValueError(
            "Unsupported LoRA adapter combination for dynamic runtime LoRA: PEFT supports "
            "only one loaded adapter with bias != 'none'."
        )

    expected_applied_state = _validate_lora_adapter_state(
        model,
        path,
        bias=bias,
        target_names=target_names,
        target_parameter_names=target_parameter_names,
        peft_config=peft_config,
        adapter_name=adapter_name,
    )
    # PEFT's state loader queries every existing TrainableTokensWrapper for the new
    # adapter. For an ordinary LoRA this raises a KeyError after adapter injection;
    # fail before mutation, but only after payload validation has had a chance to
    # reject cross-adapter trainable-token namespace spoofing precisely.
    if _has_trainable_tokens_wrapper(model):
        raise ValueError(
            "Unsupported LoRA adapter combination for dynamic runtime LoRA: an existing "
            "TrainableTokensWrapper cannot safely accept another adapter in PEFT 0.18.1."
        )
    return LoraAdapterPreflight(
        bias=bias,
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
