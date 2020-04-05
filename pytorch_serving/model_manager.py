# coding=utf-8

from guniflask.context import service
from pytorch_serving.pathutils import PathUtils
from pytorch_serving.resource_manager import ResourceManager
from pytorch_serving.PytorchOnnxService import PytorchOnnxInferenceService
from pytorch_serving.PytorchService import PytorchInferenceService
import torch
import torch.onnx
import json

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
    def save_model(self, model_name, pytorch_model, metadata, version, platform):
        """
        1. save & replace
        2. load / reload
        """
        model_path = self.path_utils.model_data_dir(model_name)
        if platform is "torch":
            self._do_save_model(model_path + "/" + str(version), pytorch_model, metadata)
        # elif platform is "torch_onnx":
        #    self._change_pytorch_model_to_onnx(model_path+ "/" + model_name, pytorch_model, metadata)
        self.load_models(model_name,model_path, metadata,version,platform)

    def delete_model(self):
        """
        1. unload
        2. delete
        """
        pass

    def load_models(self,model_name,model_path, version, platform):
        """
        load one by one
        """
        if platform == "torch":
            self.ModelServer[model_name] = PytorchInferenceService(model_name, model_path, self.resource_manager.cuda_recommendation(),version)
        if platform == "torch_onnx":
            if model_name in self.ModelServer.keys():
                self.ModelServer[model_name].dynamic_load(model_path + "/" + str(version) + ".onnx",self.resource_manager.cuda_recommendation(),version)
            else :
                self.ModelServer[model_name] = PytorchOnnxInferenceService(model_name, model_path,
                                                                   self.resource_manager.cuda_recommendation(), [str(version)])
    def model_inference(self, model_name, version, data):
        if self.ModelServer.get(model_name, None) is None :
            return "Serving is not loaded the model, please check the model name"
        return self.ModelServer[model_name].inference(version,data)

    def _change_pytorch_model_to_onnx(self,model_path,pytorch_model,metadata):
        pytorch_model.eval()
        model_input = []
        for x in metadata["input"]:
            model_input.append(torch.tensor(x))
        model_input = tuple(model_input)
        # Export the model
        print(model_input)
        torch.onnx.export(pytorch_model,  # model being run
                          model_input,  # model input (or a tuple for multiple inputs)
                          model_path+".onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=metadata["input_names"],  # the model's input names
                          output_names=metadata["output_names"],  # the model's output names
                          )
        json.dump(metadata,open(model_path+".meta","w",encoding="utf-8"))


    def _do_save_model(self,model_path,pytorch_model,metadata):
        pytorch_model.eval()
        torch.save(pytorch_model, model_path + ".pt")
        json.dump(metadata, open(model_path + ".meta", "w", encoding="utf-8"))

    def _do_load_model(self):
        pass

    def _do_delete_model(self):
        pass
