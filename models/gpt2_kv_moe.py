'''
GPT-2 model with Key-Value Memory and Mixture of Experts (MoE) architecture.
'''

import torch 
import torch.nn as nn
import time
import argparse

from layers import MultiHeadAttentionKVMOE, FeedForward, MoEFeedForward, LayerNorm
from tokenizer import Tokenizer

__all__ = ["GPTModel", "TransformerBlock"]

MOE_FF_TIME_MS = []
MOE_FF_MEM_BYTES = []


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        #  KV cache-related
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionKVMOE(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = MoEFeedForward(cfg) if cfg["num_experts"] > 0 else FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()
        start = time.perf_counter()
        x = self.ff(x)
        if use_cuda:
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated()
            MOE_FF_MEM_BYTES.append(peak_mem - base_mem)
        MOE_FF_TIME_MS.append((time.perf_counter() - start) * 1000.0)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings
    batch_size, base_len = idx.shape
    total_len = base_len + max_new_tokens
    generated = torch.empty(
        batch_size, total_len, dtype=idx.dtype, device=idx.device
    )
    generated[:, :base_len] = idx
    cur_len = base_len
    use_cuda = torch.cuda.is_available()
    MOE_FF_TIME_MS.clear()
    MOE_FF_MEM_BYTES.clear()

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            prompt_start = max(0, cur_len - ctx_len)
            logits = model(generated[:, prompt_start:cur_len], use_cache=True)

            if use_cuda:
                torch.cuda.synchronize()

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1)
                # b) append it to the running sequence (in-place)
                generated[:, cur_len] = next_idx
                cur_len += 1
                # c) feed model only the new token
                logits = model(generated[:, cur_len - 1 : cur_len], use_cache=True)

                if use_cuda:
                    torch.cuda.synchronize()
        else:
            if use_cuda:
                torch.cuda.synchronize()

            for _ in range(max_new_tokens):
                start_ctx = max(0, cur_len - ctx_len)
                logits = model(generated[:, start_ctx:cur_len], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1)
                generated[:, cur_len] = next_idx
                cur_len += 1

                if use_cuda:
                    torch.cuda.synchronize()

    if MOE_FF_TIME_MS:
        avg_ffn_time = sum(MOE_FF_TIME_MS) / len(MOE_FF_TIME_MS)
        print(f"Avg MoE FF time/call: {avg_ffn_time:.3f} ms")
    if MOE_FF_MEM_BYTES:
        avg_ffn_mem = sum(MOE_FF_MEM_BYTES) / len(MOE_FF_MEM_BYTES)
        max_ffn_mem = max(MOE_FF_MEM_BYTES)

        def to_mb(bytes_val):
            return bytes_val / (1024 ** 2)
        print(f"Avg MoE FF mem delta/call: {to_mb(avg_ffn_mem):.2f} MB (max {to_mb(max_ffn_mem):.2f} MB)")

    return generated[:, :cur_len]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=768*4, help="Intermediate FFN or MoE size.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable KV caching during generation.",
    )

    parser.add_argument(
        "--num_experts",
        type=int,
        default=0,
        help="Number of experts. If 0, use dense FFN. If >0, use MoE.",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=2,
        help="Top-k experts per token when using MoE (ignored if num_experts=0).",
    )

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = Tokenizer("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,            # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        "emb_dim": args.emb_dim,        # Embedding dimension
        "hidden_dim": args.hidden_dim,  # Intermediate size
        "n_heads": args.n_heads,        # Number of attention heads
        "n_layers": args.n_layers,      # Number of layers
        "drop_rate": 0.0,               # Dropout rate
        "qkv_bias": False,              # Query-Key-Value bias
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok if args.num_experts > 0 else 0,
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
        use_cache=not args.no_kv_cache,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()