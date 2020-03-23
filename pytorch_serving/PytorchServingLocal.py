# coding=utf-8

"""
Restful 接口
"""

from flask import request, abort, jsonify
from guniflask.web import blueprint, post_route
from pytorch_serving.model_manager import ModelManager
from pytorch_serving.resource_manager import ResourceManager
import torch

class PytorchServing:
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.model_manager = ModelManager(self.resource_manager)
        self.model_manager = self.model_manager

    def submit_model(self,data):
        """
        1. 模型的metadata
        2. 模型文件

        multipart
        """

        metadata = data["meta_data"]
        pytorch_model = data["pytorch_model"]
        model_name = data["model_name"]
        self.model_manager.save_model(model_name, pytorch_model, metadata)
        return "success!"

    def inference(self,data):
        """
        1. 模型的名称
        2. 模型的输入

        json
        """

        model_name = data["model_name"]
        model_input = data["data"]

        result = self.model_manager.model_inference(model_name,model_input)

        return result
