import torch
import requests
import torch.nn as nn
import torch.onnx
import os
import sys
from datetime import datetime
import time
import numpy as np
import random
from pytorch_serving.PytorchServingLocal import PytorchServing

class MyTest(nn.Module):
    def __init__(self, opt):
        super(MyTest, self).__init__()
        self.opt = opt
    def forward(self, A, B, C):
        ans = A+B-C
        return ans


if __name__ == '__main__':
    A = torch.tensor([6,7])
    B = torch.tensor([7,7])
    C = torch.tensor([8,7])
    test_model = MyTest({"cjx":"yj"})
    print(test_model(A,B,C))
    torch.save(test_model, "save.pt")
    torch_model = torch.load("save.pt")
    ans = torch_model(A,B,C)
    print(ans)
    x = (A,B,C)
    print(x,type(x))

    """
    torch_model.eval()
    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "MyTest.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=["A","B","C"],  # the model's input names
                      output_names=['out'],  # the model's output names
                      )
    """

    serving = PytorchServing()
    data = {
        "model_name": "test_submit",
        "pytorch_model": torch_model,
        "meta_data" : {
            "input" : ([6,7],[7,7],[8,7]),
            "input_names" : ["A","B","C"],
            "output_names" : ["output"]
        }
    }
    print(serving.submit_model(data))






"""
simple_tensorflow_serving --model_config_file="./config.json"

simple_tensorflow_serving --model_base_path="./MyTest.onnx" --model_platform="pytorch_onnx"


curl http://localhost:8500/v1/models/torch_onnx/gen_client?language=python > client.py

curl http://localhost:8500/v1/models/torch_onnx/gen_json

curl http://localhost:8500/v1/models/torch_onnx

curl http://localhost:8500/v1/models/torch_onnx:predict -X POST \
-d '{"inputs": [[1], [3]]}'
"""