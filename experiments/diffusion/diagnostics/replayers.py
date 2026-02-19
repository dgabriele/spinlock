"""Operator registry and factory for roundtrip fidelity diagnostics.

New operator types: add one entry to _REGISTRY, nothing else changes.
"""

import importlib
import logging
from typing import Any, Dict, Optional, Tuple

from experiments.diffusion.diagnostics.config import ReplayerConfig

logger = logging.getLogger(__name__)


# Registry: type string → (module_path, class_name)
_REGISTRY: Dict[str, Tuple[str, str]] = {
    "cno": ("spinlock.mno.cno_replay", "CNOReplayer"),
    "qbm": ("spinlock.qbm.replayer", "QBMReplayer"),
}


def detect_operator_type(tokenizer: Any) -> str:
    """Derive operator type from VQTokenizer's stored metadata.

    Strategy (tried in order):
    1. tokenizer.feature_metadata.families['temporal'].kept_feature_names
       — if any name contains 'quantum', infer QBM.
    2. tokenizer.config metadata dict (if training recorded operator_type).
    3. Fail with a helpful error pointing to operator_type_override.

    Args:
        tokenizer: Loaded VQTokenizer instance.

    Returns:
        Operator type string: 'qbm' or 'cno'.

    Raises:
        ValueError: If auto-detection fails (set ReplayerConfig.operator_type_override).
    """
    # Strategy 1: examine temporal feature names for physics-domain signals
    try:
        if (
            tokenizer.feature_metadata is not None
            and 'temporal' in tokenizer.feature_metadata.families
        ):
            names = tokenizer.feature_metadata.families['temporal'].kept_feature_names
            # QBM-specific extractors produce features with 'quantum' in the name
            if any('quantum' in n.lower() for n in names):
                logger.info("Auto-detected operator_type='qbm' from quantum feature names")
                return 'qbm'
            # If no quantum features, assume CNO (classical field)
            if names:
                logger.info(
                    "Auto-detected operator_type='cno' (no quantum features found in "
                    f"{len(names)} temporal feature names)"
                )
                return 'cno'
    except Exception as e:
        logger.debug(f"Feature-name heuristic failed: {e}")

    # Strategy 2: look for operator_type in tokenizer config's metadata
    try:
        meta = getattr(tokenizer, '_checkpoint_metadata', None)
        if meta and 'operator_type' in meta:
            op_type = meta['operator_type']
            logger.info(f"Auto-detected operator_type='{op_type}' from checkpoint metadata")
            return op_type
    except Exception as e:
        logger.debug(f"Metadata strategy failed: {e}")

    raise ValueError(
        "Cannot auto-detect operator type from tokenizer checkpoint.\n"
        "Set 'replayer.operator_type_override: \"cno\"' (or '\"qbm\"') in the config."
    )


def load_replayer(
    config: ReplayerConfig,
    tokenizer: Any,
    device: str,
) -> Any:
    """Instantiate the correct replayer from the registry.

    Operator type is auto-detected from the tokenizer's feature metadata.
    Falls back to config.operator_type_override if auto-detection fails.

    Args:
        config: ReplayerConfig with config_path and optional operator_type_override.
        tokenizer: Loaded VQTokenizer instance.
        device: PyTorch device string.

    Returns:
        Replayer instance (CNOReplayer or QBMReplayer).

    Raises:
        ValueError: If operator type is not in the registry.
    """
    if config.operator_type_override is not None:
        op_type = config.operator_type_override
        logger.info(f"Using operator_type_override='{op_type}'")
    else:
        op_type = detect_operator_type(tokenizer)

    if op_type not in _REGISTRY:
        raise ValueError(
            f"Unknown operator type '{op_type}'. "
            f"Registered types: {list(_REGISTRY)}"
        )

    module_path, class_name = _REGISTRY[op_type]
    cls = getattr(importlib.import_module(module_path), class_name)

    replayer = cls.from_config(str(config.config_path), device=device)
    logger.info(f"Loaded {class_name} from {config.config_path}")
    return replayer


def get_replayer_type_name(replayer: Any) -> str:
    """Return a short type string for a replayer instance."""
    class_name = type(replayer).__name__
    for op_type, (_, cn) in _REGISTRY.items():
        if cn == class_name:
            return op_type
    return class_name.lower()
