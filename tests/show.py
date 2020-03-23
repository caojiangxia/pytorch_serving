import onnx
import caffe2.python.onnx.backend as backend

# Load the ONNX model
model = onnx.load("storage/_models/test_submit/test_submit.onnx")
print(model)
# Check that the IR is well formed
onnx.checker.check_model(model)

print(model)
# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

signature = {"inputs": [], "outputs": []}
for input in model.graph.input:
    if isinstance(input, str):
        # maybe old version onnx proto
        signature["inputs"].append({"name": input})
    else:
        info = input.type.tensor_type
        input_meta = {
            "name": input.name,
            "dtype": int(info.elem_type),
            "shape": [(d.dim_value if d.HasField("dim_value") else -1)
                      for d in info.shape.dim]
        }
        signature["inputs"].append(input_meta)
    for output in model.graph.output:
        if isinstance(output, str):
            # maybe old version onnx proto
            signature["outputs"].append({"name": output})
        else:
            info = output.type.tensor_type
            output_meta = {
                "name": output.name,
                "dtype": int(info.elem_type),
                "shape": [(d.dim_value if d.HasField("dim_value") else -1)
                          for d in info.shape.dim]
            }
            signature["outputs"].append(output_meta)

print(signature)
# Build model executor
executor = backend.prepare(model)
