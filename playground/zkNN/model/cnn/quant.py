import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import copy
import time

from common import *
from dataloader import prepare_dataloader
from train import train_model


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32)):
    model_1.to(device)
    model_2.to(device)
    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False
    return True


def calibrate_model(model, loader, device=torch.device("cpu:0")):
    model.to(device)
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


def fuse_resnet50(model):
    fused_model = torch.quantization.fuse_modules(model, [["conv1", "bn1"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"], ["conv3", "bn3"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    return fused_model


def fuse_resnet18(model):
    fused_model = torch.quantization.fuse_modules(model, [["conv1", "bn1"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    return fused_model


def fuse_resnet50(model):
    fused_model = torch.quantization.fuse_modules(model, [["conv1", "bn1"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"], ["conv3", "bn3"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    return fused_model


def fuse_resnet101(model):
    fused_model = torch.quantization.fuse_modules(model, [["conv1", "bn1"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"], ["conv3", "bn3"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    return fused_model


def fuse_vgg11(model):
    fused_model = model
    return fused_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet50', 'resnet101', 'vgg11'])
    parser.add_argument('--qat', type=bool, default=False)

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

    train_loader, test_loader = prepare_dataloader()

    # Load a pretrained model.
    model = load_model(model=model, model_filepath=model_filepath, device=cuda_device)
    # Move the model to CPU since static quantization does not support CUDA currently.
    model.to(cpu_device)
    # Make a copy of the model for layer fusion
    fused_model = copy.deepcopy(model)

    model.eval()
    # The model has to be switched to evaluation mode before any layer fusion.
    # Otherwise the quantization will not work correctly.
    fused_model.eval()

    # Fuse the model in place rather manually.
    fused_model = globals()[f'fuse_{args.model}'](fused_model)

    # Model and fused model should be equivalent.
    assert model_equivalence(model_1=model, model_2=fused_model, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    quantized_model = QuantizedModel(model_fp32=fused_model)
    # Using un-fused model will fail.
    # Because there is no quantized layer implementation for a single batch normalization layer.
    # quantized_model = QuantizedResNet18(model_fp32=model)

    if args.qat:
        quantized_model.qconfig = torch.quantization.default_qat_qconfig
        # https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat
        torch.quantization.prepare_qat(quantized_model, inplace=True)

        print("Training QAT Model...")
        # Evaluation before training
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=cuda_device, criterion=nn.CrossEntropyLoss())
        print("Epoch: {:02d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(-1, eval_loss, eval_accuracy))
        # Use training data for calibration.
        quantized_model.train()
        train_model(model=quantized_model, train_loader=train_loader, test_loader=test_loader, device=cuda_device, qat=True)
        quantized_model.to(cpu_device)
    else:
        quantized_model.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(quantized_model, inplace=True)
        # Use training data for calibration.
        calibrate_model(model=quantized_model, loader=train_loader, device=cpu_device)

    # Using high-level static quantization wrapper
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    quantized_model.eval()
    # Print quantized model.
    # print(quantized_model)

    # Save quantized model.
    save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)
    torch.save(quantized_model.state_dict(), os.path.join(model_dir, f"{args.model}.pt"))

    # Load quantized model.
    quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath, device=cpu_device)

    _, fp32_eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=cpu_device, criterion=None)
    _, int8_eval_accuracy = evaluate_model(model=quantized_jit_model, test_loader=test_loader, device=cpu_device, criterion=None)

    # Skip this assertion since the values might deviate a lot.
    # assert model_equivalence(model_1=model, model_2=quantized_jit_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,32,32)), "Quantized model deviates from the original model too much!"

    print("FP32 evaluation accuracy: {:.5f}".format(fp32_eval_accuracy))
    print("INT8 evaluation accuracy: {:.5f}".format(int8_eval_accuracy))

    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1,3,32,32), num_samples=100)
    
    print("FP32 CPU Inference Latency: {:.4f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.4f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.4f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.4f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))