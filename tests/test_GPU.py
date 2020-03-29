from pytorch_serving.resource_manager import ResourceManager


if __name__ == '__main__':
    res = ResourceManager()
    print("cpu:")
    print(res.get_cpu_info())
    print("gpu:")
    print(res.get_gpu_info())
    print("recommendation:")
    id = res.cuda_recommendation()