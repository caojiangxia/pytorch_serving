import requests
import numpy
endpoint = "http://192.168.124.51:8066/caojiangxia/hello"
input_data = {
  "model_name" : "torch_onnx",
  "model_version": 1,
  "data" :  {
    "input_node_1": [1,1,1],
    "input_node_2":[9,9,18]
  }
}
result = requests.post(endpoint, json=input_data).json()

# result = requests.get(endpoint).json()

print(result)