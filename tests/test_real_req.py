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
import requests


def load_model():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 2,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_1():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 1,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_2():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 3,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_3():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 5,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def show():
    endpoint = "http://192.168.124.51:8066/api/show"

    req = {

    }

    result = requests.post(endpoint, json=req).json()

    print(result)

def inference():
    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": 1,
            "input_node_2": 2
        }

    req = {
        "model_name": "test_submit",
        "version": 2,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)


def inference_1():
    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": [1,2,3],
            "input_node_2": [4,5,6]
        }

    req = {
        "model_name": "test_submit",
        "version": 1,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_2():
    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": [3],
            "input_node_2": [4]
        }

    req = {
        "model_name": "test_submit",
        "version": 3,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_3():
    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": [3],
            "input_node_2": [4]
        }

    req = {
        "model_name": "test_submit",
        "version": 5,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

if __name__ == '__main__':

    # load_model() # 2
    # show()
    # inference()

    # load_model_1() # 1
    # inference_1()

    # load_model_2() # 3
    # inference_2()

    load_model_3() # 5
    inference_3()
# bash bin/manage debug