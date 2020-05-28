import torch
import requests
import torch.nn as nn
import torch.onnx
import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import requests


def load_model():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 2,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_1():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 1,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_2():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 3,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_3():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 5,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_4():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 7,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_5():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 8,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_6():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 10,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)


def load_model_lstm():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 9,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)


def load_model_Bowen():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "Bowen",
        "version": 3,
        "platform": "torch"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)


def load_model_jida():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "jida",
        "version": 1,
        "platform": "tensorflow"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def show():
    endpoint = "http://192.168.124.51:8066/api/show"

    req = {

    }

    result = requests.post(endpoint, json=req).json()

    print(result)

def inference():
    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": 1,
            "input_node_2": 2
        }

    req = {
        "model_name": "test_submit",
        "version": 2,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)


def inference_1():
    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": [1,2,3],
            "input_node_2": [4,5,6]
        }

    req = {
        "model_name": "test_submit",
        "version": 1,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_2():
    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": [3],
            "input_node_2": [4]
        }

    req = {
        "model_name": "test_submit",
        "version": 3,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_3():
    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": [3],
            "input_node_2": [4]
        }

    req = {
        "model_name": "test_submit",
        "version": 5,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_4():

    """
    A = torch.FloatTensor([[1,1]])
    B = torch.FloatTensor([[3,3]])
    C = torch.FloatTensor([[5,5]])
    D = torch.FloatTensor([[7,7]])

    :return:
    """

    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": [7,7]
        }
    req = {
        "model_name": "test_submit",
        "version": 7,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)


def inference_5():

    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": [[7,7,8,9],[3,4,5,6]]
        }
    req = {
        "model_name": "test_submit",
        "version": 8,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_6():

    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": [5.0]
        }
    req = {
        "model_name": "test_submit",
        "version": 10,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)


def load_model_Bowen():
    endpoint = "http://192.168.124.51:8066/api/load_model"

    req = {
        "model_name": "Bowen",
        "version": 3,
        "platform": "torch"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def inference_Bowen():
    endpoint = "http://192.168.124.51:8066/api/inference"
    data = "我是测试文本"
    req = {
        "model_name": "Bowen",
        "version": "3",
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_jida():
    endpoint = "http://192.168.124.51:8066/api/inference"
    data = '发布 单位 国务院 发布 文号 国 发 2009 31 号 发布 日期 2009 08 24 生效 日期 2009 08 24 失效 日期 所属 类别 国家 法律法规 文件 来源 中国政府 网 国务院 进一步 深化 化肥 流通 体制改革 决定 国 发 2009 31 号 各省 自治区 直辖市 人民政府 国务院 各部委 直属机构 1998 年 以来 地区 部门 认真 贯彻落实 国务院 深化 化肥 流通 体制改革 通知 国 发 1998 39 号 精神 积极 稳妥 推进 化肥 流通 体制改革 化肥 产业 得到 持续 快速 发 展 进一步 深化 化肥 流通 体制改革 调动 方面 参与 化肥 经营 积极性 不断 提高 农服 务 水平 满足 农业 生产 发展 需要 现 做出 如下 决定 放开 化肥 经营 限制 取消 化肥 经营 企业 所有制 性质 限制 允许 具备条件 所有制 组织 类型 企业 农 民 专业 合作社 个体 工商户 市场主体 进入 化肥 流通领域 参与 经营 公平竞争 申请 从事 化肥 经营 企业 相应 住所 申请 从事 化肥 经营 个体 工商户 相应 经营场所 企 业 注册资本 金 个体 工商户 资金 数额 不得 少于 3 万元 人民币 申请 省域 范围 内 设立 分支机构 从事 化肥 经营 企业 企业 总部 注册资本 金 不得 少于 1000 万元 人民币 申 请 跨省 域 设立 分支机构 从事 化肥 经营 企业 企业 总部 注册资本 金 不得 少于 3000 万 元 人民币 满足 注册资本 金 资金 数额 条件 企业 个体 工商户 直接 当地 工商行 政 管理 部门 申请 办理 登记 从事 化肥 经营 业务 企业 从事 化肥 连锁 经营 可持 企业 总部 连锁 经营 相关 文件 登记 材料 直接 门店 所在地 工商行政 管理 部门 申请 办理 登记手续 二 规范 企业 经营 行为 化肥 经营者 应 建立 进货 验收 制度 索证 索票 制度 进货 台账 销售 台账 制度 相关 记录 必须 保存 化肥 销售 后 两年 以备 查验 化肥 经营 应 明码标价 化肥 包装 标识 符合 法 律 法规 规定 国家标准 化肥 生产 经营者 不得 化肥 中 掺杂 掺假 以假充真 以次充好 不 合格 商品 冒充 合格 商品 化肥 经营者 销售 化肥 质量 负责 销售 时应 主动 出具 质量保证 证明 化肥 存在 质量 问题 消费者 质量保证 证明 依法 销售者 索赔 '
    req = {
        "model_name": "jida",
        "version": "1",
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_lstm():

    endpoint = "http://192.168.124.51:8066/api/inference"
    data = {
            "input_node_1": [3,4,2]
        }
    req = {
        "model_name": "test_submit",
        "version": 9,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_fullconnect():


    endpoint = "http://192.168.124.25:8066/api/inference"
    data = {
            "input_node_1": [[1, 1]],
            "input_node_2": [[3, 3]]
        }
    req = {
        "model_name": "FullConnect",
        "version": 1,
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

if __name__ == '__main__':

    # load_model() # 2
    # show()
    # inference()

    # load_model_1() # 1
    # inference_1()

    # load_model_2() # 3
    # inference_2()

    # load_model_3() # 5
    # inference_3()

    # load_model_4()  # 7
    # inference_4()

    # load_model_5()  # 8
    # inference_5()

    # load_model_6()  # 10
    # inference_6()

    # load_model_lstm()  # 8
    # inference_lstm()

    # load_model_Bowen()
    # show()
    # load_model_Bowen()
    # inference_Bowen()

    # load_model_jida()
    # show()
    # load_model_Bowen()
    inference_jida()
# bash bin/manage debug


# ASDASDASD

