import numpy as np
import json
import sys
import os

from utils import load_encoder_hparams_and_params


def gelu(x):
    x = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    return x

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return x

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = g * (x - mean) / np.sqrt(variance + eps) + b
    return x

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for idx, block in enumerate(blocks):
        x = transformer_block(x, **block, n_head=n_head)
    
    x = layer_norm(x, **ln_f)
    x = x @ wte.T
    return x

def load(path):
    with open(path) as f:
        sf = float(next(f))
        ndim = int(next(f))
        shape = [int(x) for x in next(f).split()]
        data = np.array([[int(x) for x in line.split()] for line in f]) * sf
        return data.reshape(shape)

def cosine_similarity(a, b):
    dot_product = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

n_tokens_to_generate = 1
model_size = "124M"
models_dir = "models"
dump_path = "../dump/gpt2"

if not os.path.exists(dump_path):
    print("Please dump first.")
    exit(1)


encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

inputs = load(os.path.join(dump_path, "input")).astype(np.int32)

fp_result = gpt2(inputs.tolist(), **params, n_head=12)
quant_result = load(os.path.join(dump_path, "final_out"))

similarity = cosine_similarity(fp_result, quant_result)
print(similarity)
