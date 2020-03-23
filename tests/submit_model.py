import torch
import torch.nn as nn
import torch.onnx
import os
import sys
from datetime import datetime
import time
import numpy as np
import random


class Trainer(object):
    def __init__(self):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


class MyTest(nn.Module):
    def __init__(self):
        super(MyTest, self).__init__()

    def forward(self, inp):
        A = inp[0]
        B = inp[1]
        C = inp[2]
        return A + C


class MyTestTrainer(Trainer):
    def __init__(self,opt):
        self.model = MyTest()
        self.opt = opt

    def update(self, batch):
        return self.model(batch)



class MyTest2(nn.Module):
    def __init__(self):
        super(MyTest2, self).__init__()

    def forward(self, A, B):
        A = torch.tensor(A)
        B = torch.tensor(B)
        ans = A + B*2
        return ans


if __name__ == '__main__':

    test  = MyTest2()
    print(test([3],[5]))

    torch.save(test, "save.pt")

    torch_model = torch.load("save.pt")
    ans = torch_model([3],[5])
    print(ans)

    A = torch.LongTensor([3])
    B = torch.LongTensor([5])
    # A = [3]
    # B = [5]
    ans = torch_model(A, B)

    print("xx:",ans)
    x = (A,B)
    torch_model.eval()
    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "MyTest.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['A',"B"],  # the model's input names
                      output_names=['output'],  # the model's output names
                      )

    """
    Trainer = MyTestTrainer({"cjx_test":"my_test"})
    Trainer.save("test.pt")

    x = (torch.tensor([6]), torch.tensor([7]), torch.tensor([8]))
    ans = Trainer.update(x)
    print(ans)

    # Input to the model
    x = (torch.tensor([6]),torch.tensor([7]),torch.tensor([8]))

    torch.save(Trainer.model, "save.pt")

    torch_model = torch.load("save.pt")
    ans = torch_model(x)
    print(ans)

    torch_model.eval()
    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "MyTest.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['cjx'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      )
    """

"""
simple_tensorflow_serving --model_config_file="./config.json"

simple_tensorflow_serving --model_base_path="./MyTest.onnx" --model_platform="pytorch_onnx"


curl http://localhost:8500/v1/models/torch_onnx/gen_client?language=python > client.py

curl http://localhost:8500/v1/models/torch_onnx/gen_json

curl http://localhost:8500/v1/models/torch_onnx

curl http://localhost:8500/v1/models/torch_onnx:predict -X POST \
-d '{"inputs": [[1], [3]]}'
"""