# coding=utf-8

"""
Restful 接口
"""

from flask import request, abort, jsonify
from guniflask.web import blueprint, post_route
from pytorch_serving.model_manager import ModelManager
from pytorch_serving.resource_manager import ResourceManager
from pytorch_serving.pathutils import PathUtils
import torch

class PytorchServing:
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.model_manager = ModelManager(self.resource_manager)
        self.path_utils = PathUtils()

    def submit_model(self,data):
        """
        1. 模型的metadata
        2. 模型文件

        multipart
        """

        model_name = data["model_name"]
        pytorch_model = data["pytorch_model"]
        metadata = data["meta_data"]
        version = data["version"]
        platform = data["platform"]
        self.model_manager.save_model(model_name, pytorch_model, metadata, version, platform)
        return "success!"

    def load_model(self,data):
        model_name = data["model_name"]
        version = data["version"]
        platform = data["platform"]
        model_path = self.path_utils.model_data_dir(model_name)
        self.model_manager.load_models(model_name, model_path, version, platform)
        return "success!"

    def inference(self,data):
        """
        1. 模型的名称
        2. 模型的输入

        json
        """

        model_name = data["model_name"]
        version  = data["version"]
        model_input = data["data"]

        result = self.model_manager.model_inference(model_name,version,model_input)

        return result
