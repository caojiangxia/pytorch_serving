import torch
import torch.nn as nn
import torch.nn.functional as F
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

class MyTest4(nn.Module):
    def __init__(self):
        super(MyTest4, self).__init__()
        self.fc1 = nn.Linear(2,100)
        self.fc2 = nn.Linear(100, 1000)
        self.fc3 = nn.Linear(1000, 1)
    def forward(self, A):
        A = self.fc1(A)
        A = F.relu(A)
        A = self.fc2(A)
        A = F.relu(A)
        A = self.fc3(A)
        return A

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


def www():
    class MyTest4(nn.Module):
        def __init__(self):
            super(MyTest4, self).__init__()
            self.fc1 = nn.Linear(2, 100)
            self.fc2 = nn.Linear(100, 1000)
            self.fc3 = nn.Linear(1000, 1)

        def forward(self, A):
            A = self.fc1(A)
            A = F.relu(A)
            A = self.fc2(A)
            A = F.relu(A)
            A = self.fc3(A)
            return A
    test = MyTest4()
    test.eval()

    A = torch.FloatTensor([[1,1]])
    B = torch.FloatTensor([[3,3]])
    C = torch.FloatTensor([[5,5]])
    D = torch.FloatTensor([[7,7]])

    print(test(A))
    print(test(B))
    print(test(C))
    print(test(D))


    torch.save(test, "save.pt")

    torch_model = torch.load("save.pt")
    ans = torch_model(A)
    print(ans)

    ans = torch_model(B)
    print(ans)

    torch_model.eval()
    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      A,  # 输入
                      "7.onnx",  # where to save the model (can be a file or file-like object)
                      # export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=9,  # the ONNX version to export the model to
                      # do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=["input_node_1"],  # the model's input names
                      output_names=["output"],  # the model's output names
                      )

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
class FullConnect(nn.Module):
    def __init__(self):
        super(FullConnect, self).__init__()
        self.fc1 = nn.Linear(2,100)
        self.fc2 = nn.Linear(100, 1000)
        self.fc3 = nn.Linear(1000, 1)
    def forward(self, input_1, input_2):
        input_1 = self.fc1(input_1)
        input_1 = F.relu(input_1)
        input_1 = self.fc2(input_1)
        input_1 = F.relu(input_1)
        input_1 = self.fc3(input_1)

        input_2 = self.fc1(input_2)
        input_2 = F.relu(input_2)
        input_2 = self.fc2(input_2)
        input_2 = F.relu(input_2)
        input_2 = self.fc3(input_2)

        output_1 = input_1
        output_2 = input_2
        output_3 = input_1+input_2
        return output_1,output_2,output_3 # return 里面不能出现运算，需要为计算完成的节点。

def generate_onnx_model():
    test = FullConnect()
    test.eval()

    input_1 = torch.FloatTensor([[1, 1]])
    input_2 = torch.FloatTensor([[3, 3]])

    print(test(input_1,input_2))

    torch.save(test, "save.pt")
    torch_model = torch.load("save.pt")
    ans = torch_model(input_1,input_2)
    print(ans)

    torch_model.eval() # 导出前需要变为eval模式
    torch.onnx.export(torch_model, # 模型接口
                      (input_1,input_2), # 任意一组单词查询的样例
                      "7.onnx",  # 前缀为当前模型的版本号
                      opset_version=9, # 导出的ONNX协议版本
                      input_names=["input_1","input_2"],  # 把forward里面的参数名写在这里
                      output_names=["output_1","output_2","output_3"],  # # 把forward后return的结果写在这里
                      )

if __name__ == '__main__':
    generate_onnx_model()


"""

{"inputs": [{"name": "input_node_1", "dtype": 1, "shape": [1, 2]}, {"name": "fc1.weight", "dtype": 1, "shape": [100, 2]}, {"name": "fc1.bias", "dtype": 1, "shape": [100]}, {"name": "fc2.weight", "dtype": 1, "shape": [1000, 100]}, {"name": "fc2.bias", "dtype": 1, "shape": [1000]}, {"name": "fc3.weight", "dtype": 1, "shape": [1, 1000]}, {"name": "fc3.bias", "dtype": 1, "shape": [1]}], "outputs": [{"name": "output", "dtype": 1, "shape": [1, 1]}, {"name": "output", "dtype": 1, "shape": [1, 1]}, {"name": "output", "dtype": 1, "shape": [1, 1]}, {"name": "output", "dtype": 1, "shape": [1, 1]}, {"name": "output", "dtype": 1, "shape": [1, 1]}, {"name": "output", "dtype": 1, "shape": [1, 1]}, {"name": "output", "dtype": 1, "shape": [1, 1]}]}

"""