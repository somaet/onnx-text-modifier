import ose 
import onnx


model = onnx.load("../resnet18-v2-7.onnx")

input_node = ['data']
output_node = ['resnetv22_relu0_fwd']

extract = ose.extract_model(model, input_node, output_node)

onnx.save(extract, "../models.onnx")