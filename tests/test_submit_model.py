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
    def forward(self, input_data):
        print(input_data)
        A = input_data["A"]
        B = input_data["B"]
        C = input_data["C"]
        ans = A+B-C
        return ans


if __name__ == '__main__':
    A = torch.tensor([6,7])
    B = torch.tensor([7,7])
    C = torch.tensor([8,7])
    data = {
        "A" : A,
        "B": B,
        "C": C,
    }
    # test_model = MyTest({"cjx":"yj"})
    # print(test_model(data))
    # torch.save(test_model, "save.pt")
    torch_model = torch.load("save.pt")
    ans = torch_model(data)
    print(ans)

    serving = PytorchServing()

    # data["model_name"]
    # version = data["version"]
    # platform = data["platform"]
    req = {
        "model_name": "test_submit",
        "version" : 2,
        "platform" : "torch"
    }
    print(serving.load_model(req))

    # data = {
    #     #     "A": 1,
    #     #     "B": 2,
    #     #     "C": 3,
    #     # }

    req = {
        "model_name": "test_submit",
        "version": 2,
        "data" : data
    }
    print(serving.inference(req))

    input()