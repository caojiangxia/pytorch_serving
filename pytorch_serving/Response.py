# -*- coding: utf-8 -*-#
# Name:         response_body
# Description:
# Author:       lty
# Date:         2020/6/24
from flask import jsonify


# 响应代码类：
class ResponseCode(object):
    SUCCESS = 200  # 成功
    FAIL = -1  # 失败
    NOT_FOUND = 404  # 未找到相关信息
    ERROR = 500

# 响应体类：
class ResBody(object):
    """
   封装响应体
   """

    def __init__(self, data=None, code=ResponseCode.SUCCESS,
                 msg="success"):
        self._data = data
        self._msg = msg
        self._code = code

    def update(self, code=None, data=None, msg=None):
        """
       更新默认响应文本
       :param code:响应状态码
       :param data: 响应数据
       :param msg: 响应消息
       :return:
       """
        if code is not None:
            self._code = code
        if data is not None:
            self._data = data
        if msg is not None:
            self._msg = msg

    @property
    def body(self):
        """
       输出响应文本内容
       :return:
       """
        temp = self.__dict__
        temp["data"] = temp.pop("_data")
        temp["msg"] = temp.pop("_msg")
        temp["code"] = temp.pop("_code")
        return jsonify(temp)


class NotFoundException(Exception):
    def __init__(self, ErrorInfo):
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo



"""
200: 请求处理成功

500: 请求处理失败

401: 请求未认证，跳转登录页

406: 请求未授权，跳转未授权提示页
"""