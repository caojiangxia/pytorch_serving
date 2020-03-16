# coding=utf-8

"""
Restful 接口
"""

from guniflask.web import blueprint, post_route

from pytorch_serving.model_manager import ModelManager


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
        pass

    @post_route('/inference')
    def inference(self):
        """
        1. 模型的名称
        2. 模型的输入

        json
        """
        pass
