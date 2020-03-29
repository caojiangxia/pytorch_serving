# coding=utf-8

"""
Restful 接口
"""

from flask import request, abort, jsonify
from guniflask.web import blueprint, post_route
from pytorch_serving.model_manager import ModelManager
from pytorch_serving.resource_manager import ResourceManager
import torch
import paramiko

@blueprint('/api')
class PytorchServing:
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.model_manager = ModelManager(self.resource_manager)
        self.model_manager = self.model_manager

    @post_route('/submit-model')
    def submit_model(self):
        """
        1. 模型的metadata
        2. 模型文件

        multipart
        """
        # 这里需要做些调整！
        data = request.json
        metadata = data["meta_data"]
        pytorch_model = data["pytorch_model"]
        model_name = data["model_name"]
        version = data["version"]
        platform = data["platform"]

        self.model_manager.save_model(model_name, pytorch_model, metadata, version, platform)
        return jsonify("success")

    @post_route('/inference')
    def inference(self):
        """
        1. 模型的名称
        2. 模型的输入

        json
        """

        data = request.json
        model_name = data["model_name"]
        model_input = data["data"]
        version = data["version"]

        result = self.model_manager.model_inference(model_name,version,model_input)

        return jsonify(result)
