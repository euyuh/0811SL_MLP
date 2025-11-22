import torch
from baseline_float_inference import DeployedClient, DeployedServer, load_export_to_pytorch

EXPORT_PATH = "./quantized_export/split_model_quantized_8bit.pt"
device = "cpu"

client = DeployedClient().to(device)
server = DeployedServer().to(device)
client, server = load_export_to_pytorch(EXPORT_PATH, client, server)
client.eval(); server.eval()

dummy = torch.randn(1, 4096)  # [1,64*64]

# 导出 client
torch.onnx.export(
    client, dummy, "client_float.onnx",
    input_names=["input"], output_names=["out_client"],
    opset_version=13
)

# 导出 server
dummy_server = torch.randn(1, 256)
torch.onnx.export(
    server, dummy_server, "server_float.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=13
)

print("Float ONNX exported.")
