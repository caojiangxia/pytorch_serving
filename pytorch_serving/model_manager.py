# coding=utf-8

from guniflask.context import service

from pytorch_serving.pathutils import PathUtils
from pytorch_serving.resource_manager import ResourceManager


@service
class ModelManager:
    """
    模型的管理
    1. 提交新模型
    2. 删除模型
    3. 启动加载模型
    4. 模型的inference
    """

    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.path_utils = PathUtils()

    def save_model(self):
        """
        1. save & replace
        2. load / reload
        """
        pass

    def delete_model(self):
        """
        1. unload
        2. delete
        """
        pass

    def load_models(self):
        """
        load one by one
        """
        pass

    def model_inference(self):
        pass

    def _do_save_model(self):
        pass

    def _do_load_model(self):
        pass

    def _do_delete_model(self):
        pass
