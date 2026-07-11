"""Runtime compatibility patches for repository training launchers."""

try:
    from transformers.modeling_rope_utils import (
        ROPE_INIT_FUNCTIONS,
        _compute_default_rope_parameters,
    )

    # Qwen2.5-VL applies multimodal sections after base rotary parameter
    # initialization. Transformers 4.57.x validates `mrope` but omits its
    # initializer mapping in some patch releases.
    ROPE_INIT_FUNCTIONS.setdefault("mrope", _compute_default_rope_parameters)
except Exception:
    # Keep site startup safe for commands that do not import transformers.
    pass
