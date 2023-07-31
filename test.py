from __future__ import annotations
import os
import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto
import ose 

def read_txt_file_to_list(file_path):
    try:
        with open(file_path, 'r') as file:
            data_list = file.readlines()
            # 데이터에 포함된 개행문자('\n')를 제거합니다.
            data_list = [data.strip() for data in data_list]
        return data_list
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []


def _build_name2obj_dict(objs):  # type: ignore
        return {obj.name: obj for obj in objs}

model = onnx.load("./resnet18-v2-7.onnx")
node = read_txt_file_to_list("./text_onnx/node_data.txt")

extract = ose.extract_text_model(model, node)
onnx.save(extract, './hello.onnx')













'''
with open('./model_graph1.txt', 'w') as file:
    file.write(str(model.graph))
mode1 = onnx.shape_inference.infer_shapes(model)
with open('./model_graph2.txt', 'w') as file:
    file.write(str(model.graph))

with open('./model_graph3.txt', 'w') as file:
    file.write(str(model.graph.initializer))
with open('./model_graph4.txt', 'w') as file:
    file.write(str(model.graph.value_info))
with open('./model_graph6.txt', 'w') as file:
    file.write(str(_build_name2obj_dict(model.graph.initializer)))
'''





