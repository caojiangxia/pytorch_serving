# coding=utf-8

from guniflask.context import service
import GPUtil
import psutil



@service
class ResourceManager:
    """
    对GPU资源的管理
    1. 为模型加载动态选择可用GPU（选择剩余显存最大的卡）
    """

    def __init__(self):
        self.Gpus = GPUtil.getGPUs()

    def get_cpu_info(self):
        ''' :return:
        memtotal: 总内存
        memfree: 空闲内存
        memused: Linux: total - free,已使用内存
        mempercent: 已使用内存占比
        cpu: 各个CPU使用占比
        '''
        mem = psutil.virtual_memory()
        memtotal = mem.total
        memfree = mem.free
        mempercent = mem.percent
        memused = mem.used
        cpu = psutil.cpu_percent(percpu=True)

        return memtotal, memfree, memused, mempercent, cpu


    def get_gpu_info(self):
        '''
        :return:
        '''
        gpulist = []
        GPUtil.showUtilization()

        remain = 0
        recommendation_id = 0
        # 获取多个GPU的信息，存在列表里
        for gpu in self.Gpus:
            # print('gpu.id:', gpu.id)
            # print('GPU总量：', gpu.memoryTotal)
            # print('GPU使用量：', gpu.memoryUsed)
            # print('gpu使用占比:', gpu.memoryUtil * 100)
            # 按GPU逐个添加信息
            gpulist.append([gpu.id, gpu.memoryTotal, gpu.memoryUsed, gpu.memoryUtil * 100])

        for id ,gpu in enumerate(gpulist):

            if remain < gpu[1] - gpu[2] :
                remain = gpu[1] - gpu[2]
                recommendation_id = id

        return recommendation_id, remain

    def cuda_recommendation(self):
        recommendation_id, remain = self.get_gpu_info()
        # 这里还有改进的空间
        return recommendation_id