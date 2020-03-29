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

