# coding=utf-8

import os
from os.path import join
from functools import wraps

from guniflask.config import settings


def create_if_not_exists(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        path = func(*args, **kwargs)
        if isinstance(path, str):
            os.makedirs(path, exist_ok=True)
        return path

    return wrapper


class PathUtils(object):
    """
    返回业务常用路径
    """

    @property
    def home_dir(self):
        return settings['home']

    @property
    @create_if_not_exists
    def models_dir(self):
        return join(self.home_dir, "_models")

    @create_if_not_exists
    def model_data_dir(self, model_name):
        return join(self.models_dir, model_name)
