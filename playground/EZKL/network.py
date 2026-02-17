import json
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
model.eval()

dummy_input = torch.randn(1, 4)
torch.onnx.export(
    model,
    dummy_input,
    "network.onnx",
    export_params=True,
    # opset_version=10,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)
print("Model exported to network.onnx")

# Generate input.json for EZKL
input_data = torch.randn(1, 4).tolist()
with open("input.json", "w") as f:
    json.dump({"input_data": input_data}, f)
print(f"Input data written to input.json: {input_data}")
