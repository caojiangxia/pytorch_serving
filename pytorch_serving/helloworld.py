# coding=utf-8

"""
Restful 接口
"""

from flask import request, abort, jsonify
from guniflask.web import blueprint, post_route,get_route

from pytorch_serving.model_manager import ModelManager


@blueprint('/caojiangxia')
class Helloworld:
    def __init__(self):
        pass

    @post_route('/hello')
    def hello(self):
        """
        this is just test
        """
        data = request.json
        print(data)
        return jsonify(data)

    @get_route('/world')
    def world(self):
        """
        this is just test
        """
        return jsonify(["world", "caojiangxia"])
