from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="client_float.onnx",
    model_output="client_int8.onnx",
    weight_type=QuantType.QInt8
)

quantize_dynamic(
    model_input="server_float.onnx",
    model_output="server_int8.onnx",
    weight_type=QuantType.QInt8
)

print("ONNX INT8 quantization done.")
