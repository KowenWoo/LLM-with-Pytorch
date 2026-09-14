import torch
import torch.nn as nn

from models.gpt_2_simple import GPT_CONFIG_124M
from layers import GELU, FeedForward

def visualize_gelu():
    import matplotlib.pyplot as plt

    gelu, relu = GELU(), nn.ReLU()

    # Some sample data
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def test_feed_forward():
    ffn = FeedForward(GPT_CONFIG_124M)
    print(ffn.layers)

    # input shape: [batch_size, num_token, embed_size]
    x = torch.rand(2,3,768)
    out = ffn(x)
    print(out.shape)
