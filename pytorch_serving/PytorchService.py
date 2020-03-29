import traceback
import logging
import os
import time
import json
import numpy as np
import torch
from guniflask.context import service

@service
class PytorchInferenceService:

    def __init__(self, model_name, model_base_path, use_cuda=False):
        """
        :param model_name:
        :param model_base_path:
        :param use_cuda:
        :param verbose:
        """
        self.model_name = model_name
        self.model_base_path = model_base_path
        self.platform = "torch"

        # Find available models
        self.model_path_list = []
        self.model_version_list = []
        self.model_metadata = []
        self.model_dict = {}
        if os.path.isdir(self.model_base_path):
            for filename in os.listdir(self.model_base_path):
                if filename.endswith(".pt"):
                    path = os.path.join(self.model_base_path, filename)

                    logging.info("Found pytorch model: {}".format(path))
                    print("Found pytorch model: {}".format(path))
                    self.model_path_list.append(path)
                    self.model_version_list.append(filename[:-3])
            if len(self.model_path_list) == 0:
                logging.error("No pytorch model found in {}".format(self.model_base_path))
        elif os.path.isfile(self.model_base_path):
            logging.info("Found pytorch model: {}".format(self.model_base_path))
            self.model_path_list.append(self.model_base_path)
        else:
            raise Exception("Invalid model_base_path: {}".format(self.model_base_path))

        # Load available models
        for id, model_path in enumerate(self.model_path_list):
            try:
                model, metadata = self.load_model(model_path,use_cuda)
                version = self.model_version_list[id]
                print("Load pytorch model: {}".format(
                    model_path))
                self.model_version_dict[version] = id # 这里应该保证相同版本号的模型只有一个才可以的
                self.model_dict[version] = model # 根据版本号找模型 和 根据版本号找id
                self.model_version_list.append(version)
                self.model_metadata.append(metadata)
            except Exception as e:
                traceback.print_exc()

    def load_model(self,model_path,use_cuda):
        model = torch.load(model_path)
        metadata = json.load(open(model_path[:-3] + ".meta","w",encoding="ust-8"))
        if use_cuda is not False:
            torch.cuda.set_device(use_cuda)
            model.cuda()
        metadata["device"] = use_cuda
        return model, metadata

    def match_metadata(self,input_data, metadata):
        """
        这里稍微有一些没有确定的地方。。。。主要在于输入的长度可能是变动的，这个时候我们是不是在metadata里面用-1来表示不定长比较好一些？
        :param input_data:
        :param metadata:
        :return:
        """
        return 1
    def inference(self, version, json_data):
        # Get version
        model_version = str(version)
        if model_version.strip() == "":
            model_version = self.model_version_list[-1]
        if str(model_version) not in self.model_version_dict:
            logging.error("No model version: {} to serve".format(model_version))
        else:
            logging.debug("Inference with json data: {}".format(json_data))
        input_data = json_data.get("data", "") # 这里面只能是list，不是tensor。

        if self.match_metadata(input_data, self.model_metadata[self.model_version_list[model_version]]) is 0:
            logging.error("input_data is not match with metadata! input_data:{}, metadata:{}".format(input_data,self.model_metadata[self.model_version_list[model_version]]))

        executor = self.model_dict[model_version]
        outputs = executor(input_data)
        return outputs # 需要注意的是这里的输出最好是list格式的，不能为tensor，要不然在jsonify的时候会容易报错。这个到时候就是我们的写的规范了。