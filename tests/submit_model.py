import torch
import torch.nn as nn
import torch.onnx
import os
import sys
from datetime import datetime
import time
import numpy as np
import random

class MyTest(nn.Module):
    def __init__(self):
        super(MyTest, self).__init__()

    def forward(self, inp ,inp2):
        A = inp
        B = inp2
        ans = A + B
        return ans

class MyTest2(nn.Module):
    def __init__(self):
        super(MyTest2, self).__init__()

    def forward(self, A, B):
        ans = A + B*5
        return ans

class MyTest3(nn.Module):
    def __init__(self):
        super(MyTest3, self).__init__()

    def forward(self, A, B):
        D = A.tolist()
        DD = B.tolist()


        D[0] += 5
        DD[1] += 10

        ans = D+DD
        ans = torch.tensor(ans)
        return ans

def qqq():
    A = torch.LongTensor([3])
    B = torch.LongTensor([5])
    test  = MyTest2()
    print(test(A,B))

    torch.save(test, "save.pt")

    torch_model = torch.load("save.pt")
    ans = torch_model(A,B)
    print(ans)

    ans = torch_model(A, B)
    print(ans)
    torch_model.eval()
    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      (A,B),  # model input (or a tuple for multiple inputs)
                      "3.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=9,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=["input_node_1","input_node_2"],  # the model's input names
                      output_names=["output"],  # the model's output names
                      )

def qq():
    test = MyTest()
    A = torch.tensor([1, 2, 3])
    B = torch.tensor([4, 5, 6])
    print(test(A,B))

    torch.save(test, "save.pt")

    torch_model = torch.load("save.pt")
    ans = torch_model(A,B)
    print(ans)

    ans = torch_model(A,B)
    print(ans)

    torch_model.eval()
    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      (A,B),  # model input (or a tuple for multiple inputs)
                      "1.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=9,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=["input_node_1","input_node_2"],  # the model's input names
                      output_names=["output"],  # the model's output names
                      )

def ww():
    test = MyTest3()
    A = torch.tensor([1,3])
    B = torch.tensor([4,9])
    print(test(A, B))

    torch.save(test, "save.pt")

    torch_model = torch.load("save.pt")
    ans = torch_model(A, B)
    print(ans)

    ans = torch_model(A, B)
    print(ans)

    torch_model.eval()
    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      (A, B),  # model input (or a tuple for multiple inputs)
                      "5.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=9,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=["input_node_1", "input_node_2"],  # the model's input names
                      output_names=["output"],  # the model's output names
                      )

if __name__ == '__main__':
    # qq()
    # qqq()
    ww()