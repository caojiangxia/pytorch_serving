import onnx
import caffe2.python.onnx.backend as backend
import numpy as np
import json
# Load the ONNX model
# model = onnx.load("storage/_models/test_submit/test_submit.onnx")
model = onnx.load("7.onnx")
# Check that the IR is well formed
onnx.checker.check_model(model)


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


print("signature: {}".format(json.dumps(signature)))


# print(signature)
# Build model executor
executor = backend.prepare(model)


NUMPY_DTYPE_MAP = {}

from onnx import TensorProto as tp
NUMPY_DTYPE_MAP.update({
        tp.FLOAT: "float32",
        tp.UINT8: "uint8",
        tp.INT8: "int8",
        tp.INT32: "int32",
        tp.INT64: "int64",
        tp.DOUBLE: "float64",
        tp.UINT32: "uint32",
        tp.UINT64: "uint64"
})

input_data = {
    "input_node_1" : [[1,1]]
}



inputs_signature = signature["inputs"]
inputs = []
if isinstance(input_data, dict):
    for input_meta in inputs_signature:
        name = input_meta["name"]
        onnx_type = input_meta["dtype"]
        if name not in input_data:
            print("Cannot find input name: {}".format(name))
        else:
            data_item = input_data[name]
            if not isinstance(data_item, np.ndarray):
                data_item = np.asarray(data_item)
            if onnx_type in NUMPY_DTYPE_MAP:
                numpy_type = NUMPY_DTYPE_MAP[onnx_type]
                if numpy_type != data_item.dtype:
                    data_item = data_item.astype(numpy_type)
            inputs.append(data_item)
else:
    raise Exception("Invalid json input data")

outputs = executor.run(inputs)

print(outputs)