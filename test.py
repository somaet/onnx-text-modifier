from __future__ import annotations
import os
import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto

def _build_name2obj_dict(objs):  # type: ignore
        return {obj.name: obj for obj in objs}

model = onnx.load("../mnist_12.onnx")

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



'''

with open('./model_graph6.txt', 'w') as file:
    file.write(str(_build_name2obj_dict(model.graph.initializer)))





