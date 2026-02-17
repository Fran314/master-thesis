import argparse
import torch
import torch.nn as nn
import numpy
import numpy as np
import sys
import os

from dataloader import prepare_dataloader
from common import *
from quant import *

def update_conv(conv, data):
    input_s = data.q_scale()
    cs = conv.weight().q_scale()
    bw = torch.quantize_per_tensor(conv.bias(), input_s * cs, 0, torch.qint32)
    bw = input_s * cs * bw.int_repr()
    conv.set_weight_bias(conv.weight(), bw)

def dump_number(f, num):
    num = 0 if num is None else num
    f.write(f"{num:.60e}\n")

def dump(filename, x=None, sf_in1=None, sf_in2=None, sf_out=None, is_fc_weight=False):
    with open(os.path.join(dump_dir, filename), "w") as f:
        dump_number(f, sf_in1)
        dump_number(f, sf_in2)
        dump_number(f, sf_out)
        if x is not None:
            if is_fc_weight:
                x = torch.transpose(x, 0, 1)
            f.write(f"{x.q_scale():.60e} {x.q_zero_point()}\n")
            f.write(f"{x.ndim}\n")
            f.write(f"{' '.join(str(d) for d in x.shape)}\n")
            np.savetxt(f, x.int_repr().numpy().flatten(), fmt="%d")

def dump_conv_param(filename, padding, stride):
    with open(os.path.join(dump_dir, filename), "w") as f:
        f.write(f"{padding} {stride}\n")

def get_func_cnt():
    func = sys._getframe(1).f_code.co_name
    cnt_name = f"{func}_cnt"
    if globals().get(cnt_name) is None:
        globals()[cnt_name] = 0
    else: 
        globals()[cnt_name] += 1
    cnt = globals()[cnt_name]
    return func, cnt

def conv(layer, input, is_conv = True):
    func, cnt = get_func_cnt()

    update_conv(layer, input)
    if is_conv:
        assert layer.padding[0] == layer.padding[1]
        assert layer.stride[0] == layer.stride[1]
        dump_conv_param(f"{func}_param_{cnt}", layer.padding[0], layer.stride[0])
    dump(f"{func}_weight_{cnt}", x=layer.weight(), sf_in1=input.q_scale(), sf_out=layer.scale, is_fc_weight=(not is_conv))
    b = torch.quantize_per_tensor(layer.bias(), input.q_scale() * layer.weight().q_scale(), 0, torch.qint32)
    # dump(f"{func}_bias_{cnt}", x=b)
    name = f"{func}_bias_{cnt}"
    with open(os.path.join(dump_dir, name), "w") as f:
        dump_number(f, 0)
        dump_number(f, 0)
        dump_number(f, 0)
        s = input.q_scale() * layer.weight().q_scale()
        f.write(f"{s:.60e} {0}\n")
        f.write(f"{b.ndim}\n")
        f.write(f"{' '.join(str(d) for d in b.shape)}\n")
        np.savetxt(f, b.int_repr().numpy().flatten(), fmt="%d")

    output = layer(input)
    dump(f"{func}_output_{cnt}", x=output)
    return output

def relu(layer, input):
    func, cnt = get_func_cnt()

    output = layer(input)
    dump(f"{func}_output_{cnt}", x=output)
    return output

def maxpool(layer, input):
    func, cnt = get_func_cnt()

    output = layer(input)
    dump(f"{func}_output_{cnt}", x=output)
    return output

def skip_add(layer, input1, input2):
    func, cnt = get_func_cnt()
    layer.scale *= (1 + 2 / 127)
    # output = layer.add(input1, input2)

    out1 = torch.quantize_per_tensor(input1.dequantize(), layer.scale, 0, torch.qint32)
    out2 = torch.quantize_per_tensor(input2.dequantize(), layer.scale, 0, torch.qint32)
    q_out = out1.int_repr().to(torch.int32) + out2.int_repr().to(torch.int32)
    assert(torch.all(q_out <= 127) and torch.all(q_out >= -127))
    output = torch.quantize_per_tensor(layer.scale * q_out, layer.scale, layer.zero_point, torch.quint8)
    a = torch.tensor(q_out, dtype=torch.int8)
    b = output.int_repr().to(torch.int16) - layer.zero_point
    assert(torch.all(a == b))

    dump(f"{func}_output1_{cnt}", x=out1, sf_in1=input1.q_scale(), sf_in2=input2.q_scale(), sf_out=output.q_scale())
    dump(f"{func}_output2_{cnt}", x=out2, sf_in1=input1.q_scale(), sf_in2=input2.q_scale(), sf_out=output.q_scale())
    dump(f"{func}_output_{cnt}", x=output, sf_in1=input1.q_scale(), sf_in2=input2.q_scale(), sf_out=output.q_scale())
    return output

def dump_resnet18(model, test_loader):
    model.eval()

    # Fetch one data from test dataset
    data, _ = next(iter(test_loader))

    # Quantitative model input
    input = model.quant(data)
    model = model.model_fp32
    dump("input", x=input) # Dump quantized input

    # Step-by-step execution of the forward to save intermediate calculations
    x = conv(model.conv1, input)
    x = relu(model.relu, x)
    x = maxpool(model.maxpool, x)
    for j in range(1, 5):
        layer = getattr(model, f"layer{j}")
        for i in range(2):
            if j != 1 and i == 0:
                x1 = conv(layer[i].downsample[0], x)
            else:
                x1 = x
            x = conv(layer[i].conv1, x)
            x = relu(layer[i].relu1, x)
            x = conv(layer[i].conv2, x)
            x = skip_add(layer[i].skip_add, x1, x)
            x = relu(layer[i].relu2, x)
    x = model.avgpool(x)
    dump("avg_output", x)
    x = torch.flatten(x, 1)
    conv(model.fc, x, is_conv=False)

    # We don't care about the dequantized output actually
    # output = model.dequant(x)

def dump_resnet50(model, test_loader):
    model.eval()

    # Fetch one data from test dataset
    data, _ = next(iter(test_loader))

    # Quantitative model input
    input = model.quant(data)
    model = model.model_fp32
    dump("input", x=input) # Dump quantized input

    # Step-by-step execution of the forward to save intermediate calculations
    x = conv(model.conv1, input)
    x = relu(model.relu, x)
    x = maxpool(model.maxpool, x)
    for layer_id, seq in enumerate([3, 4, 6, 3]):
        layer_id += 1
        layer = getattr(model, f"layer{layer_id}")
        for seq_id in range(seq):
            if seq_id == 0:
                x1 = conv(layer[seq_id].downsample[0], x)
            else:
                x1 = x
            x = conv(layer[seq_id].conv1, x)
            x = relu(layer[seq_id].relu1, x)
            x = conv(layer[seq_id].conv2, x)
            x = relu(layer[seq_id].relu1, x)
            x = conv(layer[seq_id].conv3, x)
            x = skip_add(layer[seq_id].skip_add, x1, x)
            x = relu(layer[seq_id].relu2, x)
    x = model.avgpool(x)
    dump("avg_output", x)
    x = torch.flatten(x, 1)
    conv(model.fc, x, is_conv=False)

    # We don't care about the dequantized output actually
    # output = model.dequant(x)

def dump_resnet101(model, test_loader):
    model.eval()

    # Fetch one data from test dataset
    data, _ = next(iter(test_loader))

    # Quantitative model input
    input = model.quant(data)
    model = model.model_fp32
    dump("input", x=input) # Dump quantized input

    # Step-by-step execution of the forward to save intermediate calculations
    x = conv(model.conv1, input)
    x = relu(model.relu, x)
    x = maxpool(model.maxpool, x)
    for layer_id, seq in enumerate([3, 4, 23, 3]):
        layer_id += 1
        layer = getattr(model, f"layer{layer_id}")
        for seq_id in range(seq):
            if seq_id == 0:
                x1 = conv(layer[seq_id].downsample[0], x)
            else:
                x1 = x
            x = conv(layer[seq_id].conv1, x)
            x = relu(layer[seq_id].relu1, x)
            x = conv(layer[seq_id].conv2, x)
            x = relu(layer[seq_id].relu1, x)
            x = conv(layer[seq_id].conv3, x)
            x = skip_add(layer[seq_id].skip_add, x1, x)
            x = relu(layer[seq_id].relu2, x)
    x = model.avgpool(x)
    dump("avg_output", x)
    x = torch.flatten(x, 1)
    conv(model.fc, x, is_conv=False)
    
    # We don't care about the dequantized output actually
    # output = model.dequant(x)



def dump_vgg11(model, test_loader):
    model.eval()

    # Fetch one data from test dataset
    data, _ = next(iter(test_loader))

    # Quantitative model input
    input = model.quant(data)
    x = model.quant(data)
    model = model.model_fp32
    dump("input", x=input) # Dump quantized input
    
    for layer in model.features:
        if isinstance(layer, nn.quantized.Conv2d):
            x = conv(layer, x)
        elif isinstance(layer, nn.ReLU):
            x = relu(layer, x)
        elif isinstance(layer, nn.MaxPool2d):
            x = maxpool(layer, x)
    x = model.avgpool(x)
    dump("avg_output", x)
    x = torch.flatten(x, 1)
    for layer in model.classifier:
        if isinstance(layer, nn.quantized.Linear):
            x = conv(layer, x, is_conv=False)
        elif isinstance(layer, nn.ReLU):
            x = relu(layer, x)
    
    # We don't care about the dequantized output actually
    # output = model.dequant(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet50', 'resnet101', 'vgg11'])

    args = parser.parse_args()
    
    random_seed = 0
    num_classes = 10
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "saved_models"
    model_filename = f"{args.model}_cifar10.pt"
    quantized_model_filename = f"{args.model}_quantized_cifar10.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    model = create_model(args.model, num_classes=num_classes)

    train_loader, test_loader = prepare_dataloader(batch_size=1)

    # Move the model to CPU since static quantization does not support CUDA currently.
    model.to(cpu_device)
    model.eval()

    # Fuse the model in place rather manually.
    fused_model = globals()[f'fuse_{args.model}'](model)

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    quantized_model = QuantizedModel(model_fp32=fused_model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    quantized_model.qconfig = quantization_config
    torch.quantization.prepare(quantized_model, inplace=True)
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    quantized_model.load_state_dict(torch.load(os.path.join(model_dir, f'{args.model}.pt')))

    global dump_dir
    dump_dir = os.path.join("../dump", args.model)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    globals()[f'dump_{args.model}'](quantized_model, test_loader)
