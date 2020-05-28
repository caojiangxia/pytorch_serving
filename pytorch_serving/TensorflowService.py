import traceback
import logging
import os
import time
import json
import numpy as np
import torch
from guniflask.context import service
import re
import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.contrib import learn
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# 剔除英文的符号
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(jisuanji_data_file, jiaotong_data_file):
    """
    加载二分类训练数据，为数据打上标签
    (X,[0,0])

    0:非公文---> [1,0]

    1:公文--->[0,1]

    (X,Y)

    """
    jisuanji_examples = list(open(jisuanji_data_file, "r", encoding='utf8').readlines())
    jisuanji_examples = [s.strip() for s in jisuanji_examples]
    jiaotong_exampless = list(open(jiaotong_data_file, "r", encoding='utf8').readlines())
    jiaotong_exampless = [s.strip() for s in jiaotong_exampless]
    x_text = jisuanji_examples + jiaotong_exampless

    # 适用于英文
    # x_text = [clean_str(sent) for sent in x_text]

    x_text = [sent for sent in x_text]
    # 定义类别标签 ，格式为one-hot的形式: y=1--->[0,1]
    positive_labels = [[0, 1] for _ in jisuanji_examples]
    # print positive_labels[1:3]
    negative_labels = [[1, 0] for _ in jiaotong_exampless]
    y = np.concatenate([positive_labels, negative_labels], 0)
    """
    print y
    [[0 1]
     [0 1]
     [0 1]
     ..., 
     [1 0]
     [1 0]
     [1 0]]
    print y.shape
    (10662, 2)
    """
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    利用迭代器从训练数据的回去某一个batch的数据
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # 每回合打乱顺序
        if shuffle:
            # 随机产生以一个乱序数组，作为数据集数组的下标
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        # 划分批次
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



class TensorflowInferenceService:

    def __init__(self, model_name, model_base_path, use_cuda=False, version_list=-1):
        """
        :param model_name:
        :param model_base_path:
        :param use_cuda:
        :param verbose:
        """
        self.model_name = model_name
        self.model_base_path = model_base_path
        self.platform = "tensorflow"

        # Find available models
        self.model_path_list = []
        self.model_version_list = []
        self.model_version2cuda = {}
        self.model_version2metadata = {}
        self.model_version2id = {}
        self.model_version2model = {}
        self.model_version2predicted = {}
        self.model_version2vocab = {}
        self.model_version2input = {}
        self.model_version2drop = {}


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
        self.dynamic_load("storage/_models/runs/1581514511/checkpoints/model-5400", use_cuda, 1)

    def dynamic_load(self, model_path, use_cuda, version):
        self.use_cuda = use_cuda
        version = str(version)
        this_model, this_vocab, this_pred, this_input, this_drop, metadata = self.load_model(model_path,use_cuda)
        print("Dynamic load tensorflow model: {}".format(
            model_path))
        if version in self.model_version_list:
            pass
        else :
            self.model_version_list.append(version)
            self.model_path_list.append(model_path)
            self.model_version2id[version] = len(self.model_version_list) - 1

        self.model_version2cuda[version]= use_cuda
        self.model_version2model[version] = this_model
        self.model_version2metadata[version] = metadata
        self.model_version2predicted[version] = this_pred
        self.model_version2vocab[version] = this_vocab
        self.model_version2input[version] = this_input
        self.model_version2drop[version] = this_drop


    def load_model(self,model_path,use_cuda):
        """
        model = torch.load(model_path)
        metadata = json.load(open(model_path[:-3] + ".meta","r",encoding="utf-8"))
        if use_cuda is not False:
            torch.cuda.set_device(use_cuda)
            model.cuda()
        metadata["device"] = use_cuda # 这里记载一下所在的gpu编号
        return model, metadata
        """

        # model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).cuda(use_cuda)
        # metadata = json.load(open(model_path[:-1] + "2.meta", "r", encoding="utf-8"))
        """文本one-hot编码"""

        checkpoint_dir = "storage/_models/runs/1581514511/checkpoints"
        allow_soft_placement = True
        log_device_placement = False

        vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        this_vocab = vocab_processor


        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=allow_soft_placement,
                log_device_placement=log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # 加载模型
                # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                # saver.restore(sess, checkpoint_file)

                # saver = tf.compat.v1.train.import_meta_graph("{}.meta".format("./runs/1581514511/checkpoints/model-30000"))
                # saver.restore(sess, "./runs/1581514511/checkpoints/model-30000")

                saver = tf.train.import_meta_graph("{}.meta".format("storage/_models/runs/1581514511/checkpoints/model-5400"))
                saver.restore(sess, "storage/_models/runs/1581514511/checkpoints/model-5400")

                input_x = graph.get_operation_by_name("input_x").outputs[0]

                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # scores = graph.get_operation_by_name("output/scores").outputs[0]
                # scores = tf.nn.softmax(scores)

                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                this_model = sess
                this_pred = predictions
                this_input = input_x
                this_drop = dropout_keep_prob
        metadata = {}
        return this_model, this_vocab, this_pred, this_input, this_drop, metadata


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
        model_version = str(version)
        if model_version.strip() == "":
            model_version = self.model_version_list[-1]
        if str(model_version) not in self.model_version_list:
            logging.error("No model version: {} to serve".format(model_version))
        else:
            logging.debug("Inference with json data: {}".format(input_data))


        if self.match_metadata(input_data, self.model_version2metadata[version]) is 0:
            logging.error("input_data is not match with metadata! input_data:{}, metadata:{}".format(input_data,self.model_metadata[self.model_version_list[model_version]]))

        x_raw = [input_data]
        x_test = np.array(list(self.model_version2vocab[version].transform(x_raw)))
        batch_size = 64
        batches = batch_iter(list(x_test), batch_size, 1, shuffle=False)

        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = self.model_version2model[version].run(self.model_version2predicted[version], {self.model_version2input[version]: x_test_batch, self.model_version2drop[version]: 1.0})
            # print(scores)
            all_predictions = np.concatenate([all_predictions, batch_predictions])
        # print(all_predictions)

        ret = []
        for i in all_predictions:
            if i > 0.5:
                ret.append("yes")
            else :
                ret.append("no")
        return ret # 需要注意的是这里的输出最好是list格式的，不能为tensor，要不然在jsonify的时候会容易报错。这个到时候就是我们的写的规范了。