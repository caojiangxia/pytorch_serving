
import traceback
import logging
import os
import time
import json
import numpy as np

import onnx
import caffe2.python.onnx.backend as backend

from collections import namedtuple
from guniflask.context import service
from pytorch_serving.AbstractService import AbstractInferenceService

# Lazy init
NUMPY_DTYPE_MAP = {}

class PytorchOnnxInferenceService(AbstractInferenceService):
    """
    The service to load ONNX model and make inference with pytorch-caffe2 backend.
    """

    def __init__(self, model_name, model_base_path, use_cuda=False, version_list=-1):
        """
        Initialize the service.

        Args:
          model_name: The name of the model.
          model_base_path: The file path of the model.
        Return:
          None
        """
        super(PytorchOnnxInferenceService, self).__init__()

        # Init onnx datatype map to numpy
        self.init_dtype_map()

        self.model_name = model_name
        self.model_base_path = model_base_path
        self.platform = "torch_onnx"
        self.use_cuda = use_cuda
        # Find available models
        self.model_path_list = []
        self.model_version_list = []
        self.model_version2id = {}
        self.model_version2model = {}
        self.model_version2signature = {}
        self.model_version2executor = {}
        self.model_version2metadata = {}

        if os.path.isdir(self.model_base_path):
            for filename in os.listdir(self.model_base_path):
                if filename.endswith(".onnx"):
                    path = os.path.join(self.model_base_path, filename)
                    if filename[:-5] in version_list :
                        logging.info("Found onnx model: {}".format(path))
                        print("Found onnx model: {}".format(path))
                        self.model_path_list.append(path)
                        self.model_version_list.append(filename[:-5])

            if len(self.model_path_list) == 0:
                logging.error("No onnx model found in {}".format(self.model_base_path))
        elif os.path.isfile(self.model_base_path):
            logging.info("Found onnx model: {}".format(self.model_base_path))
            self.model_path_list.append(self.model_base_path)
            if version_list == -1:
                print("parameter should have version label!")
            else :
                self.model_version_list.append(version_list[0])
        else:
            raise Exception("Invalid model_base_path: {}".format(self.model_base_path))
        # Load available models
        for id, model_path in enumerate(self.model_path_list):
            try:
                model, executor, signature, metadata = self.load_model(model_path)
                logging.info("Load onnx model: {}, signature: {}".format(
                    model_path, json.dumps(signature)))
                print("Load onnx model: {}, signature: {}".format(
                    model_path, json.dumps(signature)))
                current_version = self.model_version_list[id]
                self.model_version2id[current_version] = id
                self.model_version2model[current_version] = model
                self.model_version2executor[current_version] = executor
                self.model_version2signature[current_version] = signature
                self.model_version2metadata[current_version] = metadata
            except Exception as e:
                traceback.print_exc()

    def init_dtype_map(self):
        from onnx import TensorProto as tp
        global NUMPY_DTYPE_MAP
        NUMPY_DTYPE_MAP.update({
            tp.FLOAT: "float32",
            tp.UINT8: "uint8",
            tp.INT8: "int8",
            tp.INT32: "int32",
            tp.INT64: "int64",
            tp.DOUBLE: "float64",
            tp.UINT32: "uint32",
            tp.UINT64: "uint64"
        })

    def dynamic_load(self, model_path, use_cuda, version):
        self.use_cuda = use_cuda
        version = str(version)
        model, executor, signature, metadata = self.load_model(model_path)
        print("Dynamic load onnx model: {}, signature: {}".format(
            model_path, json.dumps(signature)))
        if version in self.model_version_list:
            pass
        else :
            self.model_version_list.append(version)
            self.model_path_list.append(model_path)
            self.model_version2id[version] = len(self.model_version_list) - 1
        self.model_version2model[version] = model
        self.model_version2executor[version] = executor
        self.model_version2signature[version] = signature
        self.model_version2metadata[version] = metadata

    def load_model(self, model_path):

        # Load model
        try :
            model = onnx.load(model_path)
        except :
            print("model is not load on server!!!")

        metadata = json.load(open(model_path[:-4] + "meta","r",encoding="utf-8"))
        onnx.checker.check_model(model)
        # Genetate signature
        signature = {"inputs": [], "outputs": []}
        for input in model.graph.input:
            if isinstance(input, str):
                # maybe old version onnx proto
                signature["inputs"].append({"name": input})
            else:
                info = input.type.tensor_type
                input_meta = {
                    "name": input.name,
                    "dtype": int(info.elem_type),
                    "shape": [(d.dim_value if d.HasField("dim_value") else -1)
                              for d in info.shape.dim]
                }
                signature["inputs"].append(input_meta)
        for output in model.graph.output:
            if isinstance(output, str):
                # maybe old version onnx proto
                signature["outputs"].append({"name": output})
            else:
                info = output.type.tensor_type
                output_meta = {
                    "name": output.name,
                    "dtype": int(info.elem_type),
                    "shape": [(d.dim_value if d.HasField("dim_value") else -1)
                              for d in info.shape.dim]
                }
                signature["outputs"].append(output_meta)
        # Build model executor
        executor = backend.prepare(model,device="CUDA:"+str(self.use_cuda))
        # executor = backend.prepare(model, device="CUDA:0")
        # executor = backend.prepare(model)
        return model, executor, signature, metadata

    def match_metadata(self,input_data, metadata):
        """
        这里稍微有一些没有确定的地方。。。。主要在于输入的长度可能是变动的，这个时候我们是不是在metadata里面用-1来表示不定长比较好一些？
        :param input_data:
        :param metadata:
        :return:
        """
        return 1

    def inference(self, version, input_data):
        # Get version
        model_version = version
        model_version = str(model_version)
        if model_version.strip() == "":
            model_version = self.model_version_list[-1]

        if str(model_version) not in self.model_version_list or input_data == "":
            logging.error("No model version: {} to serve".format(model_version))
            return "Fail to request the model version: {} with data: {}".format(
                model_version, input_data)
        else:
            logging.debug("Inference with json data: {}".format(input_data))

        signature = self.model_version2signature[model_version]
        inputs_signature = signature["inputs"]
        inputs = []
        if isinstance(input_data, dict):
            for input_meta in inputs_signature:
                name = input_meta["name"]
                onnx_type = input_meta["dtype"]
                if name not in input_data:
                    logging.error("Cannot find input name: {}".format(name))
                else:
                    data_item = input_data[name]
                    if not isinstance(data_item, np.ndarray):
                        data_item = np.asarray(data_item)
                    if onnx_type in NUMPY_DTYPE_MAP:
                        numpy_type = NUMPY_DTYPE_MAP[onnx_type]
                        if numpy_type != data_item.dtype:
                            data_item = data_item.astype(numpy_type)
                    inputs.append(data_item)
        else:
            raise Exception("Invalid json input data")

        start_time = time.time()
        executor = self.model_version2executor[model_version]
        outputs = executor.run(inputs)

        result = {}
        for idx, output_meta in enumerate(signature["outputs"]):
            name = output_meta["name"]
            print(outputs[idx])
            result[name] = outputs[idx].tolist()
        logging.debug("Inference result: {}".format(result))

        return result
