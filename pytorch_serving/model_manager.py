# coding=utf-8

from guniflask.context import service
from pytorch_serving.pathutils import PathUtils
from pytorch_serving.resource_manager import ResourceManager
from PytorchOnnxService import PytorchOnnxInferenceService

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
        self.ModelServer = {}
    def save_model(self, model_name, model_file, metadata):
        """
        1. save & replace
        2. load / reload
        """
        model_path = self.path_utils.model_data_dir(model_name)
        # 存一个文件？ 有些问题。。 这里少了几步.需要修改
        self.load_models(model_name,model_path, metadata)

    def delete_model(self):
        """
        1. unload
        2. delete
        """
        pass

    def load_models(self,model_name,model_path,metadata):
        """
        load one by one
        """
        self.ModelServer[model_name] = PytorchOnnxInferenceService(model_name, model_path, metadata, self.resource_manager.cuda_recommendation())

    def model_inference(self, model_name, data):
        if self.ModelServer.get(model_name, None) is None :
            return "Serving is not loaded the model, please check the model name"
        return self.ModelServer[model_name].inference(data)

    def _do_save_model(self):
        pass

    def _do_load_model(self):
        pass

    def _do_delete_model(self):
        pass
