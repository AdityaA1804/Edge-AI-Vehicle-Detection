from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic("best.onnx","best_int8.onnx",weight_type=QuantType.QInt8)
