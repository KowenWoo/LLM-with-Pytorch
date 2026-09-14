# LLM-with-Pytorch
From Raschka, Sebastian. Build A Large Language Model (From Scratch). Manning, 2024. ISBN: 978-1633437166.

Building LLM architectures with modern modifications for learning.

## Repository structure

```
.
├── layers/                    # reusable nn.Module building blocks, shared across architectures
│   ├── __init__.py            # re-exports the package's public API
│   ├── activation.py          # GELU, SwiGLU
│   ├── normalization.py       # LayerNorm
│   ├── feed_forward.py        # FeedForward (MLP block)
│   └── attention.py           # SelfAttention_v2, CausalAttention, MultiHeadAttentionWrapper
├── models/                    # architecture-specific model definitions
│   ├── __init__.py            # re-exports the package's public API
│   ├── gpt_2_simple.py        # GPTModel, TransformerBlock, GPT_CONFIG_124M
│   └── gpt2_kv_moe.py         # GPT-2 with KV memory + mixture-of-experts (in progress)
├── tokenizer.py                # Tokenizer, SimpleTokenizerV1
├── utils.py                     # shared helpers (e.g. generate_text_simple, visualization/test utilities)
└── dummy_model.py                # minimal placeholder model used to sanity-check the overall architecture shape
```

New architectures go in `models/` as their own file, composing building blocks from `layers/` where possible so components (attention, normalization, feed-forward, etc.) stay shared rather than duplicated per model.

`layers/__init__.py` and `models/__init__.py` re-export their package's classes, so consumers can write `from layers import LayerNorm` instead of reaching into the submodule. Import direction only flows one way — `layers/` has no dependency on `models/` — which keeps the package graph acyclic and avoids circular imports as more architectures are added.
