import argparse
import math
import numpy as np
import json
import os
from varname import nameof
import sys

from utils import load_encoder_hparams_and_params
from download_dataset import download

def dump_quantized(x, filename, enable=True):
    assert isinstance(x, QuantizedValue)
    if enable:
        with open(os.path.join(dump_dir, filename), 'w') as f:
            f.write(f"{x.scale_factor:.60e}\n")
            f.write(f"{x.value.ndim}\n")
            f.write(f"{' '.join(str(d) for d in x.value.shape)}\n")

            if x.value.dtype in [np.int16, np.int64]:
                fmt = "%d"
            elif x.value.dtype in [np.float32, np.float64]:
                fmt = "%.8f"
            else:
                print(x.value.dtype)
                assert(0)

            np.savetxt(f, x.value, fmt=fmt, delimiter=' ')


def dump_number(x, filename, enable=True):
    if enable:
        with open(os.path.join(dump_dir, filename), 'w') as f:
            if isinstance(x, np.int64):
                f.write(f"{x}\n")
            else:
                f.write(f"{x:.60e}\n")


def get_func_cnt(enable=True):
    if enable:
        func = sys._getframe(1).f_code.co_name
        cnt_name = f"{func}_cnt"
        if globals().get(cnt_name) is None:
            globals()[cnt_name] = 0
        else: 
            globals()[cnt_name] += 1
        cnt = globals()[cnt_name]
        return func, cnt
    else:
        return None, None


class QuantizedValue:
    def __init__(self, value: np.ndarray, scale_factor):
        self.value = value
        self.scale_factor = scale_factor

    @classmethod
    def from_real(cls, real_value: np.ndarray, scale_factor: float, int_type):
        assert real_value.dtype == np.float32 or real_value.dtype == np.float64
        if scale_factor is None:
            bits = 64 if int_type == np.int64 else 16
            scale_factor = get_scale_factor(real_value, bits)
        assert isinstance(scale_factor, np.float32)
        return cls(
            np.round(real_value / scale_factor).astype(int_type),
            scale_factor,
        )

    @classmethod
    def transpose(cls, q):
        assert isinstance(q, cls)
        return QuantizedValue(q.value.T, q.scale_factor)

    def __repr__(self) -> str:
        return f'"({self.value.shape}), {self.scale_factor}"'

    def to_real(self):
        return self.value * self.scale_factor
    
    def requant(self, sf, int_type):
        self.value = np.round(self.scale_factor / sf * self.value).astype(np.int16)
        self.scale_factor = sf


def get_scale_factor_extreme(mx: float, mn: float, bits):
    clip_size = max(abs(mx), abs(mn))
    clip_size = np.float32(clip_size)
    bits = np.float32((2 ** (bits - 1) - 1))
    # map real-value to symmetric range
    scale_factor = clip_size / bits
    return scale_factor


def get_scale_factor(param: np.ndarray, bits):
    return get_scale_factor_extreme(param.max(), param.min(), bits)


def q_linear(in1: QuantizedValue, in2: QuantizedValue, in3: QuantizedValue, sf4: float) -> QuantizedValue:
    s = np.float32(in1.scale_factor * in2.scale_factor / sf4)
    f = np.float32(1 / s)
    q = (in1.value.astype(np.int64) @ in2.value.astype(np.int64) + in3.value.astype(np.int64)) / f
    q = np.round(q).astype(np.int16)
    return QuantizedValue(q, sf4), f


def matrix_mul(in1: QuantizedValue, in2: QuantizedValue, sf3: float) -> QuantizedValue:
    s = np.float32(in1.scale_factor * in2.scale_factor / sf3)
    f = np.float32(1 / s)
    q = (in1.value.astype(np.int64) @ in2.value.astype(np.int64)) / f
    q = np.round(q).astype(np.int16)
    return QuantizedValue(q, sf3), f


def mul(in1: QuantizedValue, in2: QuantizedValue, sf3: float) -> QuantizedValue:
    q = (in1.scale_factor * in2.scale_factor / sf3) * (in1.value.astype(np.int64) * in2.value.astype(np.int64))
    q = np.round(q).astype(np.int16)
    return QuantizedValue(q, sf3)


def add(in1: QuantizedValue, in2: QuantizedValue, sf3: float) -> QuantizedValue:
    q = (in1.scale_factor / sf3) * in1.value.astype(np.int64) + (in2.scale_factor / sf3) * in2.value.astype(np.int64)
    q = np.round(q).astype(np.int16)
    return QuantizedValue(q, sf3)


def stats_q_linear(in1: QuantizedValue, in2: QuantizedValue, in3: QuantizedValue) -> QuantizedValue:
    r = (in1.scale_factor * in2.scale_factor) * (in1.value.astype(np.int64) @ in2.value.astype(np.int64) + in3.value.astype(np.int64))
    sf = get_scale_factor(r, 16)
    return q_linear(in1, in2, in3, sf)


def stats_matrix_mul(in1: QuantizedValue, in2: QuantizedValue) -> QuantizedValue:
    r = (in1.scale_factor * in2.scale_factor) * (in1.value.astype(np.int64) @ in2.value.astype(np.int64))
    sf = get_scale_factor(r, 16)
    return matrix_mul(in1, in2, sf)


def stats_mul(in1: QuantizedValue, in2: QuantizedValue) -> QuantizedValue:
    r = (in1.scale_factor * in2.scale_factor) * (in1.value.astype(np.int64) * in2.value.astype(np.int64))
    sf = get_scale_factor(r, 16)
    return mul(in1, in2, sf)


def stats_add(in1: QuantizedValue, in2: QuantizedValue) -> QuantizedValue:
    r = in1.scale_factor * in1.value.astype(np.int64) + in2.scale_factor * in2.value.astype(np.int64)
    sf = get_scale_factor(r, 16)
    return add(in1, in2, sf)


def mul_add(in1: QuantizedValue, in2: QuantizedValue, in3: QuantizedValue, sf3) -> QuantizedValue:
    s = np.float32(in1.scale_factor * in2.scale_factor / sf3)
    f = np.float32(1 / s)
    q = (in1.value.astype(np.int64) * in2.value.astype(np.int64) + in3.value.astype(np.int64)) / f
    q = np.round(q).astype(np.int16)
    return QuantizedValue(q, sf3), f


def stats_mul_add(in1: QuantizedValue, in2: QuantizedValue, in3: QuantizedValue) -> QuantizedValue:
    r = (in1.scale_factor * in2.scale_factor) * (in1.value.astype(np.int64) * in2.value.astype(np.int64) + in3.value.astype(np.int64))
    sf = get_scale_factor(r, 16)
    return mul_add(in1, in2, in3, sf)


def poly(x, e1, e2, e3):
    x1 = np.round(x * x / e1).astype(np.int64)
    x2 = np.round(x / e2).astype(np.int64)
    x3 = np.round(1 / e3).astype(np.int64)
    return x1, x2, x3, x1 + x2 + x3


def i_exp(x: QuantizedValue, enable):
    func, cnt = get_func_cnt(enable)

    assert np.all(x.value <= 0)

    a = np.float32(0.35850)
    b = np.float32(1.353)
    c = np.float32(0.344)

    ln2 = np.log(2, dtype=np.float32)

    s = np.float32(-x.scale_factor / ln2)
    f = np.float32(1 / s)
    z = np.floor(x.value / f).astype(np.int64)
    dump_number(-f, f"{func}_{nameof(z)}_f_{cnt}", enable)
    dump_quantized(QuantizedValue(z, 1), f"{func}_{nameof(z)}_{cnt}", enable)

    sp = ln2 / np.float32(32767)
    sp *= np.float32(1 + 2 / (2**15 - 1))
    s1 = np.float32(x.scale_factor / sp)
    s2 = np.float32(ln2 / sp)
    f1 = np.float32(1 / s1)
    f2 = np.float32(1 / s2)
    p1 = QuantizedValue(np.round(x.value / f1).astype(np.int64), sp)
    p2 = QuantizedValue(np.round(z / f2).astype(np.int64), sp)
    qp = np.round(p1.value + p2.value).astype(np.int64)
    p = QuantizedValue(qp, sp)
    dump_number(f1, f"{func}_p1_f_{cnt}", enable)
    dump_number(f2, f"{func}_p2_f_{cnt}", enable)
    dump_quantized(p1, f"{func}_{nameof(p1)}_{cnt}", enable)
    dump_quantized(p2, f"{func}_{nameof(p2)}_{cnt}", enable)
    dump_quantized(p, f"{func}_{nameof(p)}_{cnt}", enable)

    sl = np.float32(a * b * b + c) / np.float32(32767) # poly(0) / 32767
    sl *= np.float32(1 + 2 / (2**15 - 1))
    e1 = np.float32(a * sp * sp) / sl
    f1 = np.float32(1 / e1)
    e2 = np.float32(2 * a * b * sp) / sl
    f2 = np.float32(1 / e2)
    e3 = np.float32(a * b * b + c) / sl
    f3 = np.float32(1 / e3)
    x1, x2, x3, ql = poly(p.value, f1, f2, f3)
    dump_number(f1, f"{func}_f1_{cnt}", enable)
    dump_number(f2, f"{func}_f2_{cnt}", enable)
    dump_number(f3, f"{func}_f3_{cnt}", enable)
    dump_quantized(QuantizedValue(x1, sl), f"{func}_x1_{cnt}", enable)
    dump_quantized(QuantizedValue(x2, sl), f"{func}_x2_{cnt}", enable)
    dump_number(x3, f"{func}_x3_{cnt}", enable)
    l = QuantizedValue(ql, sl)
    dump_quantized(l, f"{func}_{nameof(l)}_{cnt}", enable)

    assert(np.all(ql >= 0))
    x = QuantizedValue(ql >> z, sl)
    dump_quantized(x, f"{func}_out_{cnt}", enable)
    return x


def i_sqrt(n):
    if n == 0:
        return 0
    x = 2 ** ((int(n).bit_length() + 1) // 2)
    while True:
        x1 = (x + n // x) // 2
        if x1 >= x:
            return x
        x = x1


def i_sqrt_ndim(n):
    return np.vectorize(i_sqrt)(n)


def i_softmax(x: QuantizedValue, enable):
    func, cnt = get_func_cnt(enable)

    dump_quantized(x, f"{func}_in_{cnt}", enable)
    # all non-linear opeartions use int64
    x.value = x.value.astype(np.int64)

    x_max = np.max(x.value, axis=-1, keepdims=True)
    dump_quantized(QuantizedValue(x_max, x.scale_factor), f"{func}_{nameof(x_max)}_{cnt}", enable)
    x.value = x.value - x_max
    assert np.all(x.value <= 0)

    x = i_exp(QuantizedValue(x.value, x.scale_factor), enable)

    # requantization to int16
    t = QuantizedValue.from_real(x.value / np.sum(x.value, axis=-1, keepdims=True), None, np.int16)
    f = np.float32(t.scale_factor)
    t = QuantizedValue(np.round(x.value / f).astype(np.int64), t.scale_factor)
    dump_number(f, f"{func}_out1_f_{cnt}", enable)
    dump_quantized(t, f"{func}_out1_{cnt}", enable)
    x = QuantizedValue(np.round(t.value / np.sum(x.value, axis=-1, keepdims=True)).astype(np.int16), t.scale_factor)
    dump_quantized(x, f"{func}_out_{cnt}", enable)
    return x


def i_erf(x: QuantizedValue):
    func, cnt = get_func_cnt()

    a = np.float32(-0.2888)
    b = np.float32(-1.769)
    c = np.float32(1)
    dump_number(a, f"{func}_{nameof(a)}_{cnt}")
    dump_number(b, f"{func}_{nameof(b)}_{cnt}")
    dump_number(c, f"{func}_{nameof(c)}_{cnt}")

    sign = np.sign(x.value)
    dump_quantized(QuantizedValue(sign, 1), f"{func}_{nameof(sign)}_{cnt}")

    q_abs = np.abs(x.value)
    dump_quantized(QuantizedValue(q_abs, 1), f"{func}_{nameof(q_abs)}_{cnt}")

    sqrt2 = np.float32(math.sqrt(2))
    dump_number(sqrt2, f"{func}_{nameof(sqrt2)}_{cnt}")

    t = np.float32(-sqrt2 * b / x.scale_factor)
    f = np.float32(1 / t)
    zb = np.floor(1 / f).astype(np.int64)
    dump_number(f, f"{func}_{nameof(zb)}_f_{cnt}")
    dump_number(zb, f"{func}_{nameof(zb)}_{cnt}")

    q_min = np.minimum(q_abs, zb).astype(np.int64)
    dump_quantized(QuantizedValue(q_min, 1), f"{func}_{nameof(q_min)}_{cnt}")

    sl = c / np.float32(32767) # poly(-b) / 32767
    sl *= np.float32(1 + 2 / (2**15 - 1))
    e1 = np.float32(a * x.scale_factor * x.scale_factor / 2) / sl
    f1 = np.float32(1 / e1)
    e2 = np.float32(2 * a * b * x.scale_factor / sqrt2) / sl
    f2 = np.float32(1 / e2)
    e3 = np.float32(a * b * b + c) / sl
    f3 = np.float32(1 / e3)
    x1, x2, x3, ql = poly(q_min, f1, f2, f3)
    dump_number(-f1, f"{func}_f1_{cnt}")
    dump_number(f2, f"{func}_f2_{cnt}")
    dump_number(f3, f"{func}_f3_{cnt}")
    dump_quantized(QuantizedValue(-x1, sl), f"{func}_x1_{cnt}")
    dump_quantized(QuantizedValue(x2, sl), f"{func}_x2_{cnt}")
    dump_number(x3, f"{func}_x3_{cnt}")
    # ql = np.round(1 * 32767 * poly(q_min * x.scale_factor / sqrt2, a, b, c)).astype(np.int64)
    l = QuantizedValue(ql, sl)
    dump_quantized(l, f"{func}_{nameof(l)}_{cnt}")

    x = QuantizedValue((l.value * sign).astype(np.int64), l.scale_factor)
    dump_quantized(x, f"{func}_out_{cnt}")
    return x


def gelu(x: QuantizedValue) -> QuantizedValue:
    func, cnt = get_func_cnt()
    dump_quantized(x, f"{func}_in_{cnt}")

    x.value = x.value.astype(np.int64)
    t = i_erf(x)
    # requantization to int16
    sf = get_scale_factor(x.scale_factor / np.float32(2) * x.value + (x.scale_factor / np.float32(2 * 32767)) * x.value * t.value, 16)
    sf *= np.float32(2)
    s1 = np.float32(x.scale_factor / np.float32(2) / sf)
    s2 = np.float32((x.scale_factor / np.float32(2 * 32767) / sf))
    f1 = np.float32(1 / s1)
    f2 = np.float32(1 / s2)
    y1 = np.round(x.value / f1).astype(np.int16)
    y2 = np.round(x.value * t.value / f2).astype(np.int16)
    y = (y1 + y2).astype(np.int16)
    y = QuantizedValue(y, sf)
    dump_number(f1, f"{func}_f1_{cnt}")
    dump_number(f2, f"{func}_f2_{cnt}")
    dump_quantized(QuantizedValue(y1, sf), f"{func}_out1_{cnt}")
    dump_quantized(QuantizedValue(y2, sf), f"{func}_out2_{cnt}")
    dump_quantized(y, f"{func}_out_{cnt}")
    return y


def softmax(x, enable) -> QuantizedValue:
    return i_softmax(x, enable)


def layer_norm(x: QuantizedValue, g, b) -> QuantizedValue:
    func, cnt = get_func_cnt()

    dump_quantized(x, f"{func}_in_{cnt}")
    x.value = x.value.astype(np.int64)
    mean = np.round(np.mean(x.value, axis=-1, keepdims=True)).astype(np.int64)
    var = np.round(np.mean((x.value - mean) ** 2, axis=-1, keepdims=True)).astype(np.int64)
    std = i_sqrt_ndim(var).astype(np.int64)
    std_max = np.maximum(1, std, dtype=np.int64)

    dump_quantized(QuantizedValue(mean, x.scale_factor), f"{func}_{nameof(mean)}_{cnt}")
    dump_quantized(QuantizedValue(var, x.scale_factor * x.scale_factor), f"{func}_{nameof(var)}_{cnt}")
    dump_quantized(QuantizedValue(std,  x.scale_factor), f"{func}_{nameof(std)}_{cnt}")
    dump_quantized(QuantizedValue(std_max,  x.scale_factor), f"{func}_{nameof(std_max)}_{cnt}")

    # can not use integer division
    # x = QuantizedValue.from_real((x.value * x.scale_factor - mean * x.scale_factor) / (i_sqrt_ndim(variance) * x.scale_factor), None, np.int16)
    t = QuantizedValue.from_real((x.value - mean) / std_max, None, np.int16)
    f = t.scale_factor
    x = QuantizedValue(np.round((x.value - mean) / f).astype(np.int64), t.scale_factor)
    dump_quantized(x, f"{func}_sub_{cnt}")
    x = QuantizedValue(np.round(x.value / std_max).astype(np.int16), t.scale_factor)
    dump_quantized(x, f"{func}_norm_{cnt}")

    g = QuantizedValue.from_real(g, None, np.int16)
    b = QuantizedValue.from_real(b, x.scale_factor * g.scale_factor, np.int64)
    dump_quantized(x, f"{func}_in_{cnt}")
    dump_quantized(g, f"{func}_{nameof(g)}_{cnt}")
    dump_quantized(b, f"{func}_{nameof(b)}_{cnt}")

    x, f = stats_mul_add(x, g, b)
    dump_number(f, f"{func}_out_f_{cnt}")
    dump_quantized(x, f"{func}_out_{cnt}")
    return x


def linear(x: QuantizedValue, w, b) -> QuantizedValue:
    func, cnt = get_func_cnt()

    w = QuantizedValue.from_real(w, None, np.int16)
    b = QuantizedValue.from_real(b, x.scale_factor * w.scale_factor, np.int64)
    dump_quantized(x, f"{func}_in_{cnt}")
    dump_quantized(w, f"{func}_{nameof(w)}_{cnt}")
    dump_quantized(b, f"{func}_{nameof(b)}_{cnt}")

    # not set sf3 in stats phase
    # buf get sf3 after calculating, and quantize calculate result by sf3
    x, f = stats_q_linear(x, w, b)
    dump_number(f, f"{func}_f_{cnt}")
    dump_quantized(x, f"{func}_out_{cnt}")
    return x


def ffn(x, c_fc, c_proj) -> QuantizedValue:
    return linear(gelu(linear(x, **c_fc)), **c_proj)


def attention(q: QuantizedValue, k: QuantizedValue, v: QuantizedValue, sf) -> QuantizedValue:
    enable = sf is not None

    func, cnt = get_func_cnt(enable)
    
    k = QuantizedValue.transpose(k)

    dump_quantized(q, f"{func}_{nameof(q)}_{cnt}", enable)
    dump_quantized(k, f"{func}_{nameof(k)}_{cnt}", enable)
    dump_quantized(v, f"{func}_{nameof(v)}_{cnt}", enable)

    # q * k.T
    x, f = stats_matrix_mul(q, k)
    dump_number(f, f"{func}_qk_f_{cnt}", enable)
    dump_quantized(x, f"{func}_qk_{cnt}", enable)
    # q * k.T / sqrt(d)
    sqrt_dim = np.sqrt(q.value.shape[-1], dtype=np.float32)
    s1 = x.scale_factor
    t = QuantizedValue.from_real(x.to_real() / sqrt_dim, None, np.int16)
    s = np.float32(s1 / sqrt_dim / t.scale_factor)
    f = np.float32(1 / s)
    x = QuantizedValue(np.round(t.value / f).astype(np.int16), t.scale_factor)
    dump_number(f, f"{func}_divqk_f_{cnt}", enable)
    dump_quantized(x, f"{func}_divqk_{cnt}", enable)
    # softmax
    x = softmax(x, enable)
    # @ v
    if sf is None:
        x, f = stats_matrix_mul(x, v)
    else:
        x, f = matrix_mul(x, v, sf)
        dump_number(f, f"{func}_qkv_f_{cnt}", enable)
    dump_quantized(x, f"{func}_qkv_{cnt}", enable)
    return x


def mha(x: QuantizedValue, c_attn, c_proj, n_head) -> QuantizedValue:
    func, cnt = get_func_cnt()

    x = linear(x, **c_attn)
    qkv = np.split(x.value, 3, axis=-1)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))

    # get scaling factor of all attention output
    out_heads = []
    for q, k, v in zip(*qkv_heads):
        q = QuantizedValue(q, x.scale_factor)
        k = QuantizedValue(k, x.scale_factor)
        v = QuantizedValue(v, x.scale_factor)
        attn = attention(q, k, v, None)
        out_heads.append(attn.to_real())
    t = np.hstack(out_heads)
    t = QuantizedValue.from_real(t, None, np.int16)

    # requantize and stack
    out_heads = []
    for q, k, v in zip(*qkv_heads):
        q = QuantizedValue(q, x.scale_factor)
        k = QuantizedValue(k, x.scale_factor)
        v = QuantizedValue(v, x.scale_factor)
        attn = attention(q, k, v, t.scale_factor)
        assert(attn.scale_factor == t.scale_factor)
        out_heads.append(attn.value)
    x = QuantizedValue(np.hstack(out_heads), t.scale_factor)
    dump_quantized(x, f"{func}_mha_{cnt}", x)

    x = linear(x, **c_proj)
    return x

def res(input1, input2):
    func, cnt = get_func_cnt()
    # dump_quantized(input1, f"{func}_in1_{cnt}")
    # dump_quantized(input2, f"{func}_in2_{cnt}")
    x = stats_add(input1, input2)
    sf = np.float32(x.scale_factor * (1 + 2 / (2**15 - 1)))
    f1 = 1 / (input1.scale_factor / sf)
    f2 = 1 / (input2.scale_factor / sf)
    out1 = QuantizedValue(np.round(input1.value / f1).astype(np.int64), sf)
    out2 = QuantizedValue(np.round(input2.value / f2).astype(np.int64), sf)
    q_out = out1.value + out2.value
    x = QuantizedValue(q_out, sf)

    dump_number(f1, f"{func}_f1_{cnt}")
    dump_number(f2, f"{func}_f2_{cnt}")
    dump_quantized(out1, f"{func}_out1_{cnt}")
    dump_quantized(out2, f"{func}_out2_{cnt}")
    dump_quantized(x, f"{func}_out_{cnt}")
    return x

def transformer_block(x: QuantizedValue, mlp, attn, ln_1, ln_2, n_head) -> QuantizedValue:
    x1 = mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = res(x, x1)
    x1 = ffn(layer_norm(x, **ln_2), **mlp)
    x = res(x, x1)
    return x


def gpt2(input, wte, wpe, blocks, ln_f, n_head) -> QuantizedValue:
    dump_quantized(QuantizedValue(np.array(input, np.int64), 1), nameof(input))

    wte = QuantizedValue.from_real(wte, None, np.int16)
    wpe = QuantizedValue.from_real(wpe, None, np.int16)
    # dump_quantized(wte, nameof(wte))
    # dump_quantized(wpe, nameof(wpe))

    x = stats_add(QuantizedValue(wte.value[input], wte.scale_factor), QuantizedValue(wpe.value[range(len(input))], wpe.scale_factor))
    dump_quantized(x, "embd_out")

    # scaling factor of x should be max(abs(max(wte, wpe)), abs(min(wte, wpe)))
    # sf = max(abs(max(wte.value.max(), wpe.value.max())), abs(min(wte.value.min(), wpe.value.min())))
    # x = QuantizedValue(x, sf)

    for idx, block in enumerate(blocks):
        x = transformer_block(x, **block, n_head=n_head)

    x = layer_norm(x, **ln_f)
    x, f = stats_matrix_mul(x, QuantizedValue.transpose(wte))
    dump_quantized(x, "final_out")
    return x


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head).value
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))

    return inputs[len(inputs) - n_tokens_to_generate :]


def main(length, idx):
    n_tokens_to_generate = 1
    model_size = "124M"
    models_dir = "models"
    dataset = "./data/small-117M.test.jsonl"
    dump_path = "../dump/gpt2"

    if not os.path.exists(dataset):
        download('small-117M')

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    global dump_dir
    dump_dir = dump_path

    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    with open(dataset, "r") as f:
        for line in f.readlines():
            line = json.loads(line)
            if line["length"] == length:
                idx -= 1
                if idx != 0:
                    continue
                prompt = line["text"]
                input_ids = encoder.encode(prompt)

                # if len(input_ids) + n_tokens_to_generate >= hparams["n_ctx"]:
                #     continue

                output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

                # output_text = encoder.decode(output_ids)
                # print(output_text)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--idx", type=int, default=1)
    args = vars(parser.parse_args())

    main(**args)
