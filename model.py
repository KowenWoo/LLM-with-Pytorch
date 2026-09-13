import torch
import torch.nn as nn
import tiktoken

from attention_mechanisms import MultiHeadAttentionWrapper

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # size supported by BPE tokenizer
    "context_length": 1024, # max input token count
    "emb_dim": 768,         # embedding dimension
    "n_heads": 12,          # number of attention heads
    "n_layers": 12,         # number of layers
    "drop_rate": 0.1,       # dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionWrapper(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # add the original input back

        # shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # add the original input back

        return x


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            )
        
    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale =- nn.Parameter(torch.ones(emb_dim)) # learnable scale parameter
        self.bias = nn.Parameter(torch.zeros(emb_dim)) # learnable bias parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # approximation of the cumulative distribution function of the standard normal distribution
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3)))
        )

class Tokenizer():
    def __init__(self, type):
        self.tokenizer = tiktoken.get_encoding(type)

    def encode(self, input):
        return torch.tensor(self.tokenizer.encode(input))



if __name__ == "__main__":

    #tokenize inputs
    tokenizer = Tokenizer("gpt2")

    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(tokenizer.encode(txt1))
    batch.append(tokenizer.encode(txt2))
    batch = torch.stack(batch, dim=0)
    print(batch)


    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)  

    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4

    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f"Total size of the model: {total_size_mb:.2f} MB")


    
