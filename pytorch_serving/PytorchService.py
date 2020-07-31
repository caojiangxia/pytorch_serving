import traceback
import logging
import os
import time
import json
import numpy as np
import torch
from guniflask.context import service
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_serving.Gongwen import GongwenTokenizer
import pytorch_serving.Desc.utils.tokenizer as Desctokenizer
import pytorch_serving.Desc.utils.score as Descscore
import pytorch_serving.Desc.utils.torch_utils as Desctorch_utils
from pytorch_serving.Desc.model import RelationModel as DescRelationModel
import requests
class PytorchInferenceService:
    def __init__(self, model_name, model_base_path, use_cuda=False, version_list=-1):
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
        self.model_version2cuda = {}
        self.model_version2metadata = {}
        self.model_version2id = {}
        self.model_version2model = {}
        self.model_version2Tokenizer = {}

        """
        if os.path.isdir(self.model_base_path):
            for filename in os.listdir(self.model_base_path):
                if filename.endswith(".pt"):
                    path = os.path.join(self.model_base_path, filename)
                    if filename[:-3] in version_list:
                        logging.info("Found pytorch model: {}".format(path))
                        print("Found pytorch model: {}".format(path))
                        self.model_path_list.append(path)
                        self.model_version_list.append(filename[:-3])
            if len(self.model_path_list) == 0:
                logging.error("No pytorch model found in {}".format(self.model_base_path))
        elif os.path.isfile(self.model_base_path):
            logging.info("Found pytorch model: {}".format(self.model_base_path))
            self.model_path_list.append(self.model_base_path) # 版本号有一些问题？
            if version_list == -1:
                print("parameter should have version label!")
            else :
                self.model_version_list.append(version_list[0])
        else:
            raise Exception("Invalid model_base_path: {}".format(self.model_base_path))

        # Load available models 像这里是一次导入所有的模型，其实是不太好的。
        for id, model_path in enumerate(self.model_path_list):
            try:
                model, metadata = self.load_model(model_path,use_cuda)
                current_version = self.model_version_list[id]
                print("Load pytorch model: {}".format(
                    model_path))
                self.model_version2id[current_version] = id # 这里应该保证相同版本号的模型只有一个才可以的
                self.model_version2model[current_version] = model # 根据版本号找模型 和 根据版本号找id
                self.model_version2metadata.append(metadata)
                self.model_version2cuda.append(use_cuda)
            except Exception as e:
                traceback.print_exc()
        """

        print(model_base_path + "/" + version_list[0], "torch_loading")
        self.dynamic_load(model_base_path + "/" + version_list[0] ,use_cuda, version_list[0])

    def dynamic_load(self, model_path, use_cuda, version ):
        self.use_cuda = use_cuda
        version = str(version)

        if self.model_name == "Bowen" :
            model, tokenizer, metadata = self.Bowen_load_model(model_path,use_cuda)
        if self.model_name == "Description" or self.model_name == "Synonym" or self.model_name == "Hyponym":
            model, tokenizer, metadata = self.Desc_load_model(model_path, use_cuda)

        print("Dynamic load pytorch model: {}".format(
            model_path))
        if version in self.model_version_list:
            pass
        else :
            self.model_version_list.append(version)
            self.model_path_list.append(model_path)
            self.model_version2id[version] = len(self.model_version_list) - 1

        self.model_version2cuda[version]= use_cuda
        self.model_version2model[version] = model
        self.model_version2metadata[version] = metadata
        self.model_version2Tokenizer[version] = tokenizer


    def Bowen_load_model(self,model_path,use_cuda):
        """
        model = torch.load(model_path)
        metadata = json.load(open(model_path[:-3] + ".meta","r",encoding="utf-8"))
        if use_cuda is not False:
            torch.cuda.set_device(use_cuda)
            model.cuda()
        metadata["device"] = use_cuda # 这里记载一下所在的gpu编号
        return model, metadata
        """
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).cuda(use_cuda)
        # metadata = json.load(open(model_path[:-1] + "2.meta", "r", encoding="utf-8"))
        metadata = {}
        tokenizer = GongwenTokenizer(model_path)
        return model, tokenizer, metadata


    def Desc_load_model(self,model_path,use_cuda):

        model_file = model_path + '/best_model.pt'
        opt = Desctorch_utils.load_config(model_file) 
        if "ema" not in opt:
            opt['ema'] = 0.999
        opt['bert_model'] = '/home/kdsec/serving/storage/_models/roberta/'
        model = DescRelationModel(opt, 10)
        model.load(model_file)

        model.model.cuda(use_cuda)
        tokenizer = Desctokenizer.Tokenizer(model_path + '/vocab.txt', do_lower_case=True)
        # model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).cuda(use_cuda)
        # metadata = json.load(open(model_path[:-1] + "2.meta", "r", encoding="utf-8"))
        metadata = {}
        return model, tokenizer, metadata

    def match_metadata(self,input_data, metadata):
        """
        这里稍微有一些没有确定的地方。。。。主要在于输入的长度可能是变动的，这个时候我们是不是在metadata里面用-1来表示不定长比较好一些？
        :param input_data:
        :param metadata:
        :return:
        """
        return 1

    def inference(self, version, input_data):
        if self.model_name == "Bowen":
            # Get version
            model_version = str(version)
            if model_version.strip() == "":
                model_version = self.model_version_list[-1]
            if str(model_version) not in self.model_version_list:
                logging.error("No model version: {} to serve".format(model_version))
            else:
                logging.debug("Inference with json data: {}".format(input_data))


            if self.match_metadata(input_data, self.model_version2metadata[version]) is 0:
                logging.error("input_data is not match with metadata! input_data:{}, metadata:{}".format(input_data,self.model_metadata[self.model_version_list[model_version]]))

            executor = self.model_version2model[model_version]

            input_data = self.model_version2Tokenizer[model_version].generate(input_data)

            all_input_ids = torch.tensor(input_data["input_ids"], dtype=torch.long).cuda(self.model_version2cuda[model_version])
            all_input_mask = torch.tensor(input_data["input_mask"], dtype=torch.long).cuda(self.model_version2cuda[model_version])
            all_segment_ids = torch.tensor(input_data["segment_ids"], dtype=torch.long).cuda(self.model_version2cuda[model_version])

            logits = executor(all_input_ids,all_segment_ids,all_input_mask)

            preds = []
            preds.append(logits.detach().cpu().numpy())
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            output = preds.tolist()
            ret = []
            for i in output:
                if i == 0 :
                    ret.append("yes")
                else :
                    ret.append("no")
            return ret # 需要注意的是这里的输出最好是list格式的，不能为tensor，要不然在jsonify的时候会容易报错。这个到时候就是我们的写的规范了。

        if self.model_name == "Description" or self.model_name == "Synonym" or self.model_name == "Hyponym":
            print(input_data)
            subject = input_data['subject']
            sentence_list = input_data['sentences']
            batch_size = 8
            data = [sentence_list[i:i + batch_size] for i in range(0, len(sentence_list), batch_size)]
            results = []

            model_version = str(version)
            if model_version.strip() == "":
                model_version = self.model_version_list[-1]

            if str(model_version) not in self.model_version_list:
                logging.error("No model version: {} to serve".format(model_version))
            else:
                logging.debug("Inference with subject: {}".format(subject))

            if self.match_metadata(input_data, self.model_version2metadata[version]) is 0:
                logging.error("input_data is not match with metadata! input_data:{}, metadata:{}".format(input_data,
                                                                                                         self.model_metadata[
                                                                                                             self.model_version_list[
                                                                                                            model_version]]))
            executor = self.model_version2model[model_version]

            for sents in data:
                results.extend(
                    Descscore.extract_items(sents, subject, self.model_version2Tokenizer[model_version], executor, user_cuda = self.model_version2cuda[model_version]))


            res = {}
            res["res"] = results
            res["keyword"] = input_data['subject']
            res["expand_token"] = ""
            res["model_name"] = self.model_name

            if "callback" in input_data:
                callback_url = input_data['callback']
                requests.post(callback_url,json = res,headers = {"Content-Type": "application/json"})

            return results
