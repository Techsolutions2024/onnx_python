import onnx

# Load mô hình
onnx_model = onnx.load("models\LPDNet_usa_pruned_tao5.onnx")

# Lấy thông tin input
print("=== INPUTS ===")
for input_tensor in onnx_model.graph.input:
    tensor_type = input_tensor.type.tensor_type
    shape = []
    for d in tensor_type.shape.dim:
        shape.append(d.dim_value if d.dim_value > 0 else "None")
    print(f"Name: {input_tensor.name}, Shape: {shape}, DataType: {tensor_type.elem_type}")

# Lấy thông tin output
print("\n=== OUTPUTS ===")
for output_tensor in onnx_model.graph.output:
    tensor_type = output_tensor.type.tensor_type
    shape = []
    for d in tensor_type.shape.dim:
        shape.append(d.dim_value if d.dim_value > 0 else "None")
    print(f"Name: {output_tensor.name}, Shape: {shape}, DataType: {tensor_type.elem_type}")
