# coding=utf-8

"""
Restful 接口
"""

from flask import request, abort, jsonify
from guniflask.web import blueprint, post_route
from pytorch_serving.model_manager import ModelManager
from pytorch_serving.resource_manager import ResourceManager
from pytorch_serving.pathutils import PathUtils
import json

from pytorch_serving.Response import ResBody, ResponseCode, NotFoundException

import torch
import requests


@blueprint('/api')
class PytorchServing:
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.model_manager = ModelManager(self.resource_manager)
        self.path_utils = PathUtils()
        self.warm_restart()

    def warm_restart(self):
        pass

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
        version = str(data["version"])
        platform = data["platform"]

        self.model_manager.save_model(model_name, pytorch_model, metadata, version, platform)
        return jsonify("success")

    @post_route('/load_model')
    def load_model(self):
        data = request.json
        model_name = data["model_name"]
        version = str(data["version"])
        platform = data["platform"]
        model_path = self.path_utils.model_data_dir(model_name)
        self.model_manager.load_models(model_name, model_path, version, platform)

        self._save_now()

        res = ResBody()
        res.update("success")
        return res.body

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
        if request.json.get('callback', None) is not None :
            model_input["callback"] = request.json.get('callback')

        version = str(data["version"])

        result = self.model_manager.model_inference(model_name,version,model_input)

        return result

    @post_route('/Description')
    def Description(self):
        """
        1. 模型的名称
        2. 模型的输入

        json
        """
        data = request.json

        data["model_name"] = "Description"
        model_name = data["model_name"]
        model_input = data["data"]
        if request.json.get('callback', None) is not None:
            model_input["callback"] = request.json.get('callback')
        version = str(data["version"])
        result = self.model_manager.model_inference(model_name, version, model_input)
        return result

    @post_route('/Synonym')
    def Synonym(self):
        """
        1. 模型的名称
        2. 模型的输入

        json
        """
        data = request.json

        data["model_name"] = "Synonym"
        model_name = data["model_name"]
        model_input = data["data"]
        if request.json.get('callback', None) is not None:
            model_input["callback"] = request.json.get('callback')
        version = str(data["version"])
        result = self.model_manager.model_inference(model_name, version, model_input)
        return result

    @post_route('/Hyponym')
    def Hyponym(self):
        """
        1. 模型的名称
        2. 模型的输入

        json
        """
        data = request.json

        data["model_name"] = "Hyponym"
        model_name = data["model_name"]
        model_input = data["data"]
        if request.json.get('callback', None) is not None:
            model_input["callback"] = request.json.get('callback')
        version = str(data["version"])
        result = self.model_manager.model_inference(model_name, version, model_input)
        return result

    @post_route('/Document-recognition')
    def jida(self):
        """
        1. 模型的名称
        2. 模型的输入

        json
        """
        data = request.json

        data["model_name"] = "jida"
        model_name = data["model_name"]
        model_input = data["data"]
        if request.json.get('callback', None) is not None:
            model_input["callback"] = request.json.get('callback')
        version = str(data["version"])
        result = self.model_manager.model_inference(model_name, version, model_input)
        return result

    @post_route('/Document-discrimination')
    def Bowen(self):
        """
        1. 模型的名称
        2. 模型的输入

        json
        """
        data = request.json

        data["model_name"] = "Bowen"
        model_name = data["model_name"]
        model_input = data["data"]
        if request.json.get('callback', None) is not None:
            model_input["callback"] = request.json.get('callback')
        version = str(data["version"])
        result = self.model_manager.model_inference(model_name, version, model_input)
        return result

    @post_route('/show')
    def show(self):
        """
        1. 查看目前加载的模型的一些信息。
        json
        """
        all_model_information = {}
        for model_name in self.model_manager.ModelServer:
            all_model_information[model_name] = self.model_manager.ModelServer[model_name].model_version_list
        res = ResBody()
        res.update(data={"result": all_model_information})
        return res.body

    def _save_now(self):
        """
        1. 存储当前的状态，方便重启的时候重新加载。
        """

        all_model_information = {}
        for model_name in self.model_manager.ModelServer:
            all_model_information[model_name] = self.model_manager.ModelServer[model_name].model_version_list

        home_path = self.path_utils.home_dir + "/status.json"
        json.dump(all_model_information,open(home_path,"w",encoding="utf-8"))
