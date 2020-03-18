# coding=utf-8

"""
Restful 接口
"""

from flask import request, abort, jsonify
from guniflask.web import blueprint, post_route
from pytorch_serving.model_manager import ModelManager
import torch

@blueprint('/api')
class PytorchServing:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    @post_route('/submit-model')
    def submit_model(self):
        """
        1. 模型的metadata
        2. 模型文件

        multipart
        """
        data = request.json
        metadata = data["meta_data"]
        model_file = data["model_file"]
        model_name = data["model_name"]
        self.model_manager.save_model(model_name, model_file, metadata)

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

        result = self.model_manager.model_inference(model_name,model_input)

        return jsonify(result)
