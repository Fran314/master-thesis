# zknn-models

This repository holds the CNN and Transformer models used by zkNN:

- ResNet: From [Pytorch](https://github.com/pytorch/vision/blob/release/0.8.0/torchvision/models/resnet.py).
  - ResNet18
  - ResNet50
  - ResNet101
- VGG11: From [Pytorch](https://github.com/pytorch/vision/blob/release/0.8.0/torchvision/models/vgg.py). The same network structure as in [zkCNN](https://github.com/TAMUCrypto/zkCNN).
- GPT-2: From [jaymody/picoGPT](https://github.com/jaymody/picoGPT)

The repository code contains three functions:

- Training: Train all CNN models from scratch.
- Quantization: Quantize CNN models using the Pytorch API, or GPT-2 using methods in [I-BERT](https://arxiv.org/abs/2101.01321).
- Dump: Dump the input, intermediate calculation results of each layer during one inference process.

## Setup
Our experiments are finished in the following setup.

- Python 3.11.5
- CUDA 12.4
- AMD Ryzen 3700X 8-Core CPU 
- NVIDIA GeForce RTX 3090 Ti

Python requirements.
```bash
pip install -r requirements.txt
```

## Dataset

For all CNN models, use [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

For GPT-2, use the dataset from [OpenAI](https://github.com/openai/gpt-2-output-dataset).

## Traning CNN Models

Training CNN model from scratch.

For example, run the following command in `./zknn/model/cnn` directory to train ResNet50:

```python
python train.py --model resnet50
```
Supported model: `{resnet18, resnet50, resnet101, vgg11}`

## Quantize CNN Models

Quantizing CNN model using the Pytorch API.

For example, in `./zknn/model/cnn` directory, quantifying  ResNet50 using Post Training Quantization (PTQ):

```python
python quant.py --model resnet50
```

Or using Quantization Aware Training (QAT):

```python
python quant.py --model resnet50 --qat
```

## Dump Quantized CNN Models

Dump CNN model input and inference intermediate calculations results.

For example, in `./zknn/model/cnn` directory, dump quantized ResNet50:

```python
python dump.py --model resnet50
```

The results are stored in the `./zknn/model/dump` directory

## Quantize and Dump Transformer Model

The parameters of the transformer model have been trained and they are provided by OpenAI. 

Quantize and dump GPT-2 with input token length 64 in `./zknn/model/transformer` directory:

```python
python quant_and_dump.py --length 64
```

The results are stored in the `./zknn/model/dump` directory

## Accuracy

| Models    | Original | Quantized PTQ | Quantized QAT |
| --------- | -------- | ------------- | ------------- |
| ResNet50  | 0.9396   | 0.9364        | 0.9400        |
| ResNet101 | 0.9383   | 0.9308        | 0.9379        |
| VGG11     | 0.9173   | 0.9170        | 0.9181        |

The cosine similarity of the outputs from our GPT-2 and the GPT-2 provided by OpenAI reaches 99.95%
