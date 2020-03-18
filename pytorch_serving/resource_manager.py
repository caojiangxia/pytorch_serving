# coding=utf-8

from guniflask.context import service


@service
class ResourceManager:
    """
    对GPU资源的管理
    1. 为模型加载动态选择可用GPU（选择剩余显存最大的卡）
    """

    def __init__(self):
        pass

    def cuda_recommendation(self):
        pass