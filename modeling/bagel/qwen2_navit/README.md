# `qwen2_navit` package reconstruction

This document describes how the former monolithic module `qwen2_navit.py` was split into a **package** (`qwen2_navit/`) for clearer ownership, reuse, and navigation. Behavior and public symbols are preserved.

## Motivation

- **Separation of concerns**: configuration, KV cache utilities, attention, decoder blocks, and top-level LM live in dedicated modules.
- **Easier navigation**: smaller files map to one main concept each.
- **Stable imports**: code that used `from modeling.bagel.qwen2_navit import …` continues to work via `qwen2_navit/__init__.py`.

## Layout

| Module | Responsibility |
|--------|------------------|
| [`config.py`](config.py) | `Qwen2Config` — extends the base Qwen2 config with Bagel-specific fields (`qk_norm`, `layer_module`, `freeze_und`, etc.). |
| [`cache.py`](cache.py) | `NaiveCache` (per-layer key/value dicts) and `BaseNavitOutputWithPast` (model output dataclass). |
| [`utils.py`](utils.py) | `pad_sequence` — pads packed tensors along the sequence dimension for flex-attention paths. |
| [`attention.py`](attention.py) | `PackedAttention` (single projection path) and `PackedAttentionMoT` (dual und/gen projections). Includes compiled `flex_attention` and training/inference attention logic. |
| [`decoder_layers.py`](decoder_layers.py) | `Qwen2DecoderLayer`, `Qwen2MoTDecoderLayer`, `Qwen2MoEDecoderLayer`, and `Decoder_layer_dict` (maps `config.layer_module` strings to layer classes). |
| [`modeling.py`](modeling.py) | `Qwen2Model` (embeddings, layer stack, RoPE, final norm) and `Qwen2ForCausalLM` (LM head and forwards). |
| [`__init__.py`](__init__.py) | Re-exports the public API listed below. |

## Dependency direction (high level)

```
config  ←  attention, decoder_layers, modeling
cache   ←  attention, decoder_layers, modeling
utils   ←  attention
attention  ←  decoder_layers
decoder_layers  ←  modeling
```

`modeling` depends on `decoder_layers` (for `Decoder_layer_dict`); it does not import `attention` directly.

## Public API (`__all__`)

The package exposes:

- **Config**: `Qwen2Config`
- **Cache / outputs**: `NaiveCache`, `BaseNavitOutputWithPast`
- **Utils**: `pad_sequence`
- **Attention**: `PackedAttention`, `PackedAttentionMoT`
- **Layers**: `Qwen2DecoderLayer`, `Qwen2MoTDecoderLayer`, `Qwen2MoEDecoderLayer`, `Decoder_layer_dict`
- **Models**: `Qwen2Model`, `Qwen2ForCausalLM`

Submodules can also be imported explicitly, e.g. `from modeling.bagel.qwen2_navit.attention import PackedAttention`.

## Migration notes

- **Removed**: single file `modeling/bagel/qwen2_navit.py` (replaced by the `qwen2_navit/` directory). In Python, the package name matches the old module name, so import paths at `modeling.bagel.qwen2_navit` stay valid.
- **Unchanged**: class names, method signatures, and checkpoint key structure for existing `Qwen2Config` / layer types.
- **External callers** (`bagel.py`, `fsdp_utils.py`, eval scripts, etc.) that import `NaiveCache`, decoder layer classes, or config/model classes require **no** import-path changes when using the package root.

## License

Files retain the original Apache-2.0 headers and upstream attributions from Qwen / Hugging Face and project-specific modifications as in the source headers.
