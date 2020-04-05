import requests
import numpy
endpoint = "http://192.168.124.51:8066/caojiangxia/hello"
input_data = {
  "model_name" : "torch_onnx",
  "model_version": 1
}
result = requests.post(endpoint, json=input_data).json()

# result = requests.get(endpoint).json()

print(result)




# bash bin/manage debug
# CUDA_VISIBLE_DEVICES=3 bash bin/manage debug