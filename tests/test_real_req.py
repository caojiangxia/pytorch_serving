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
    endpoint = "http://192.168.124.51:8166/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 2,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_1():
    endpoint = "http://192.168.124.51:8166/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 1,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_2():
    endpoint = "http://192.168.124.51:8166/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 3,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_3():
    endpoint = "http://192.168.124.51:8166/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 5,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_4():
    endpoint = "http://192.168.124.51:8166/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 7,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_5():
    endpoint = "http://192.168.124.51:8166/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 8,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_6():
    endpoint = "http://192.168.124.51:8166/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 10,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)


def load_model_lstm():
    endpoint = "http://192.168.124.51:8166/api/load_model"

    req = {
        "model_name": "test_submit",
        "version": 9,
        "platform": "torch_onnx"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)


def load_model_Bowen():
    endpoint = "http://192.168.124.51:8166/api/load_model"

    req = {
        "model_name": "Bowen",
        "version": 3,
        "platform": "torch"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)


def load_model_jida():
    endpoint = "http://192.168.126.29:8166/api/load_model"

    req = {
        "model_name": "jida",
        "version": 1,
        "platform": "tensorflow"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def show():
    endpoint = "http://192.168.126.29:8166/api/show"

    req = {

    }

    result = requests.post(endpoint, json=req).json()

    print(result)

def inference():
    endpoint = "http://192.168.126.29:8166/api/inference"
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
    endpoint = "http://192.168.124.51:8166/api/inference"
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
    endpoint = "http://192.168.124.51:8166/api/inference"
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
    endpoint = "http://192.168.124.51:8166/api/inference"
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

    endpoint = "http://192.168.124.51:8166/api/inference"
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

    endpoint = "http://192.168.124.51:8166/api/inference"
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

    endpoint = "http://192.168.124.51:8166/api/inference"
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
    endpoint = "http://192.168.126.29:8166/api/load_model"

    req = {
        "model_name": "Bowen",
        "version": 3,
        "platform": "torch"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_Desc():
    endpoint = "http://192.168.126.29:8166/api/load_model"

    req = {
        "model_name": "Description",
        "version": 1,
        "platform": "torch"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def load_model_Syno():
    endpoint = "http://192.168.126.29:8166/api/load_model"

    req = {
        "model_name": "Synonym",
        "version": 1,
        "platform": "torch"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)


def load_model_Hypo():
    endpoint = "http://192.168.126.29:8166/api/load_model"

    req = {
        "model_name": "Hyponym",
        "version": 1,
        "platform": "torch"
    }

    result = requests.post(endpoint, json=req, timeout=None).json()

    print(result)

def inference_Bowen():
    endpoint = "http://192.168.126.29:8166/api/inference"
    data = "我是测试文本"
    req = {
        "model_name": "Bowen",
        "version": "3",
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_Bowen_Bowen():
    endpoint = "http://192.168.126.29:8166/api/Document-discrimination"
    data = "我是测试文本"
    req = {
        "version": "3",
        "data" : data
    }
    """
    {
        "version": "3",
        "data" : "我是测试文本"
    }
    """
    result = requests.post(endpoint, json=req).json()
    print(result)


def inference_Desc():
    endpoint = "http://192.168.126.29:8166/api/inference"
    data = {'subject':'习近平','sentences':["▲6月8日至10日，中共中央总书记、国家主席、中央军委主席习近平在宁夏考察。","习近平总书记也是一位具有独特语言风格且善用妙喻的党的领袖。","习近平总书记作为中国巨轮的掌舵人，他的比喻背后传递的是党和国家的政治主张，展现的是治国理政的方略，宣传的是经济发展的理念，释放的是改革创新的政策信号，提供的是全球治理的中国方案。","央视网消息（新闻联播）：国家主席习近平5月28日签署了第四十五号主席令。"]}

    req = {
        "model_name": "Description",
        "version": "1",
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)
def inference_Desc_Desc():
    endpoint = "http://192.168.126.29:8166/api/Description"
    data = {'subject':'习近平','sentences':["▲6月8日至10日，中共中央总书记、国家主席、中央军委主席习近平在宁夏考察。","习近平总书记也是一位具有独特语言风格且善用妙喻的党的领袖。","习近平总书记作为中国巨轮的掌舵人，他的比喻背后传递的是党和国家的政治主张，展现的是治国理政的方略，宣传的是经济发展的理念，释放的是改革创新的政策信号，提供的是全球治理的中国方案。","央视网消息（新闻联播）：国家主席习近平5月28日签署了第四十五号主席令。"]}

    req = {
        "version": "1",
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)
def inference_Syno():
    endpoint = "http://192.168.126.29:8166/api/inference"
    data = {'subject':'美国海军','sentences':["美国海军（United States Navy，简称USN或U.S. Navy）是美利坚合众国武装力量的一个分支，负责管理所有与海军有关的事务。","它足够专业，以至于法国海军（MN）、美国海军（USN）及其他军事组织都是它的拥趸。","美国海军的前身是在美国独立战争中建立大陆海军，于1790年解散。"]}
    req = {
        "model_name": "Synonym",
        "version": "1",
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)
def inference_Syno_Syno():
    endpoint = "http://192.168.126.29:8166/api/Synonym"
    data = {'subject':'美国海军','sentences':["美国海军（United States Navy，简称USN或U.S. Navy）是美利坚合众国武装力量的一个分支，负责管理所有与海军有关的事务。","它足够专业，以至于法国海军（MN）、美国海军（USN）及其他军事组织都是它的拥趸。","美国海军的前身是在美国独立战争中建立大陆海军，于1790年解散。"]}
    req = {
        "version": "1",
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)


def inference_Hypo():
    endpoint = "http://192.168.126.29:8166/api/inference"
    data = {'subject':'敏感地区','sentences':["敏感地区的国家一般是 俄罗斯 ，迪拜 ，伊朗， 伊拉克， 阿富汗， 克罗地亚，巴哈马，安哥拉，缅甸，孟加拉国，柬埔寨，哥伦比亚，肯尼亚，哈萨克斯坦，黎巴嫩，利比亚，尼泊尔，尼日利亚，巴拉圭，秘鲁，塞尔维亚，多哥，叙利亚，墨西哥合众国等等等等。","美国防部文件曝光：美军在这个敏感地区又有大动作"]}
    req = {
        "model_name": "Hyponym",
        "version": "1",
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)
def inference_Hypo_Hypo():
    endpoint = "http://192.168.126.29:8166/api/Hyponym"
    data = {'subject':'敏感地区','sentences':["敏感地区的国家一般是 俄罗斯 ，迪拜 ，伊朗， 伊拉克， 阿富汗， 克罗地亚，巴哈马，安哥拉，缅甸，孟加拉国，柬埔寨，哥伦比亚，肯尼亚，哈萨克斯坦，黎巴嫩，利比亚，尼泊尔，尼日利亚，巴拉圭，秘鲁，塞尔维亚，多哥，叙利亚，墨西哥合众国等等等等。","美国防部文件曝光：美军在这个敏感地区又有大动作"]}
    req = {
        "version": "1",
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_jida():
    endpoint = "http://192.168.126.29:8166/api/inference"
    data = '发布 单位 国务院 发布 文号 国 发 2009 31 号 发布 日期 2009 08 24 生效 日期 2009 08 24 失效 日期 所属 类别 国家 法律法规 文件 来源 中国政府 网 国务院 进一步 深化 化肥 流通 体制改革 决定 国 发 2009 31 号 各省 自治区 直辖市 人民政府 国务院 各部委 直属机构 1998 年 以来 地区 部门 认真 贯彻落实 国务院 深化 化肥 流通 体制改革 通知 国 发 1998 39 号 精神 积极 稳妥 推进 化肥 流通 体制改革 化肥 产业 得到 持续 快速 发 展 进一步 深化 化肥 流通 体制改革 调动 方面 参与 化肥 经营 积极性 不断 提高 农服 务 水平 满足 农业 生产 发展 需要 现 做出 如下 决定 放开 化肥 经营 限制 取消 化肥 经营 企业 所有制 性质 限制 允许 具备条件 所有制 组织 类型 企业 农 民 专业 合作社 个体 工商户 市场主体 进入 化肥 流通领域 参与 经营 公平竞争 申请 从事 化肥 经营 企业 相应 住所 申请 从事 化肥 经营 个体 工商户 相应 经营场所 企 业 注册资本 金 个体 工商户 资金 数额 不得 少于 3 万元 人民币 申请 省域 范围 内 设立 分支机构 从事 化肥 经营 企业 企业 总部 注册资本 金 不得 少于 1000 万元 人民币 申 请 跨省 域 设立 分支机构 从事 化肥 经营 企业 企业 总部 注册资本 金 不得 少于 3000 万 元 人民币 满足 注册资本 金 资金 数额 条件 企业 个体 工商户 直接 当地 工商行 政 管理 部门 申请 办理 登记 从事 化肥 经营 业务 企业 从事 化肥 连锁 经营 可持 企业 总部 连锁 经营 相关 文件 登记 材料 直接 门店 所在地 工商行政 管理 部门 申请 办理 登记手续 二 规范 企业 经营 行为 化肥 经营者 应 建立 进货 验收 制度 索证 索票 制度 进货 台账 销售 台账 制度 相关 记录 必须 保存 化肥 销售 后 两年 以备 查验 化肥 经营 应 明码标价 化肥 包装 标识 符合 法 律 法规 规定 国家标准 化肥 生产 经营者 不得 化肥 中 掺杂 掺假 以假充真 以次充好 不 合格 商品 冒充 合格 商品 化肥 经营者 销售 化肥 质量 负责 销售 时应 主动 出具 质量保证 证明 化肥 存在 质量 问题 消费者 质量保证 证明 依法 销售者 索赔 '
    req = {
        "model_name": "jida",
        "version": "1",
        "data" : data
    }
    result = requests.post(endpoint, json=req).json()
    print(result)
def inference_jida_jida():
    endpoint = "http://192.168.126.29:8166/api/Document-recognition"
    data = '发布 单位 国务院 发布 文号 国 发 2009 31 号 发布 日期 2009 08 24 生效 日期 2009 08 24 失效 日期 所属 类别 国家 法律法规 文件 来源 中国政府 网 国务院 进一步 深化 化肥 流通 体制改革 决定 国 发 2009 31 号 各省 自治区 直辖市 人民政府 国务院 各部委 直属机构 1998 年 以来 地区 部门 认真 贯彻落实 国务院 深化 化肥 流通 体制改革 通知 国 发 1998 39 号 精神 积极 稳妥 推进 化肥 流通 体制改革 化肥 产业 得到 持续 快速 发 展 进一步 深化 化肥 流通 体制改革 调动 方面 参与 化肥 经营 积极性 不断 提高 农服 务 水平 满足 农业 生产 发展 需要 现 做出 如下 决定 放开 化肥 经营 限制 取消 化肥 经营 企业 所有制 性质 限制 允许 具备条件 所有制 组织 类型 企业 农 民 专业 合作社 个体 工商户 市场主体 进入 化肥 流通领域 参与 经营 公平竞争 申请 从事 化肥 经营 企业 相应 住所 申请 从事 化肥 经营 个体 工商户 相应 经营场所 企 业 注册资本 金 个体 工商户 资金 数额 不得 少于 3 万元 人民币 申请 省域 范围 内 设立 分支机构 从事 化肥 经营 企业 企业 总部 注册资本 金 不得 少于 1000 万元 人民币 申 请 跨省 域 设立 分支机构 从事 化肥 经营 企业 企业 总部 注册资本 金 不得 少于 3000 万 元 人民币 满足 注册资本 金 资金 数额 条件 企业 个体 工商户 直接 当地 工商行 政 管理 部门 申请 办理 登记 从事 化肥 经营 业务 企业 从事 化肥 连锁 经营 可持 企业 总部 连锁 经营 相关 文件 登记 材料 直接 门店 所在地 工商行政 管理 部门 申请 办理 登记手续 二 规范 企业 经营 行为 化肥 经营者 应 建立 进货 验收 制度 索证 索票 制度 进货 台账 销售 台账 制度 相关 记录 必须 保存 化肥 销售 后 两年 以备 查验 化肥 经营 应 明码标价 化肥 包装 标识 符合 法 律 法规 规定 国家标准 化肥 生产 经营者 不得 化肥 中 掺杂 掺假 以假充真 以次充好 不 合格 商品 冒充 合格 商品 化肥 经营者 销售 化肥 质量 负责 销售 时应 主动 出具 质量保证 证明 化肥 存在 质量 问题 消费者 质量保证 证明 依法 销售者 索赔 '
    req = {
        "version": "1",
        "data" : data
    }
    """
    {
        "version": "1",
        "data" : '发布 单位 国务院 发布 文号 国 发 2009 31 号 发布 日期 2009 08 24 生效 日期 2009 08 24 失效 日期 所属 类别 国家 法律法规 文件 来源 中国政府 网 国务院 进一步 深化 化肥 流通 体制改革 决定 国 发 2009 31 号 各省 自治区 直辖市 人民政府 国务院 各部委 直属机构 1998 年 以来 地区 部门 认真 贯彻落实 国务院 深化 化肥 流通 体制改革 通知 国 发 1998 39 号 精神 积极 稳妥 推进 化肥 流通 体制改革 化肥 产业 得到 持续 快速 发 展 进一步 深化 化肥 流通 体制改革 调动 方面 参与 化肥 经营 积极性 不断 提高 农服 务 水平 满足 农业 生产 发展 需要 现 做出 如下 决定 放开 化肥 经营 限制 取消 化肥 经营 企业 所有制 性质 限制 允许 具备条件 所有制 组织 类型 企业 农 民 专业 合作社 个体 工商户 市场主体 进入 化肥 流通领域 参与 经营 公平竞争 申请 从事 化肥 经营 企业 相应 住所 申请 从事 化肥 经营 个体 工商户 相应 经营场所 企 业 注册资本 金 个体 工商户 资金 数额 不得 少于 3 万元 人民币 申请 省域 范围 内 设立 分支机构 从事 化肥 经营 企业 企业 总部 注册资本 金 不得 少于 1000 万元 人民币 申 请 跨省 域 设立 分支机构 从事 化肥 经营 企业 企业 总部 注册资本 金 不得 少于 3000 万 元 人民币 满足 注册资本 金 资金 数额 条件 企业 个体 工商户 直接 当地 工商行 政 管理 部门 申请 办理 登记 从事 化肥 经营 业务 企业 从事 化肥 连锁 经营 可持 企业 总部 连锁 经营 相关 文件 登记 材料 直接 门店 所在地 工商行政 管理 部门 申请 办理 登记手续 二 规范 企业 经营 行为 化肥 经营者 应 建立 进货 验收 制度 索证 索票 制度 进货 台账 销售 台账 制度 相关 记录 必须 保存 化肥 销售 后 两年 以备 查验 化肥 经营 应 明码标价 化肥 包装 标识 符合 法 律 法规 规定 国家标准 化肥 生产 经营者 不得 化肥 中 掺杂 掺假 以假充真 以次充好 不 合格 商品 冒充 合格 商品 化肥 经营者 销售 化肥 质量 负责 销售 时应 主动 出具 质量保证 证明 化肥 存在 质量 问题 消费者 质量保证 证明 依法 销售者 索赔 '
    }
    """
    result = requests.post(endpoint, json=req).json()
    print(result)

def inference_lstm():

    endpoint = "http://192.168.124.51:8166/api/inference"
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


    endpoint = "http://192.168.124.25:8166/api/inference"
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

    """
    load_model_Bowen()
    load_model_Desc()
    load_model_jida()
    load_model_Syno()
    load_model_Hypo()
    show()
    """


    # load_model_Bowen()
    # show()
    inference_Bowen()
    inference_Bowen_Bowen()

    # load_model_Desc()
    # show()
    inference_Desc()
    inference_Desc_Desc()

    # load_model_Syno()
    # show()
    inference_Syno()
    inference_Syno_Syno()

    # load_model_Hypo()
    # show()
    inference_Hypo()
    inference_Hypo_Hypo()

    # load_model_jida()
    # show()
    inference_jida()
    inference_jida_jida()
    show()




# bash bin/manage debug




