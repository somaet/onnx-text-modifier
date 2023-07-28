import os
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import ose 

class onnxText: 
    def __init__(self, content_list):
        self.nodes = content_list
        
    @classmethod 
    def from_stream(cls, stream):
        print("load text...")
        stream.seek(0)
        content_list = [line.decode('utf-8').strip() for line in stream]
        print(content_list)
        stream.close()
        return cls(content_list)
    


class onnxDownload: 
    def __init__(self, model_proto, nodes, submodels):
        self.model_proto = model_proto
        self.modes = nodes
        self.submodels = submodels

    @classmethod 
    def from_model(cls, model_proto, nodes):
        sub_models = [] 
        for i in range(0, len(nodes), 2):
            input_nodes = [item.strip() for item in nodes[i].split(',')]
            output_nodes = [item.strip() for item in nodes[i+1].split(',')]
            extract = ose.extract_model(model_proto, input_nodes, output_nodes)
            sub_models.append(extract)
        return cls(model_proto, nodes, sub_models)
    

    def save_model(self, save_dir='./text_onnx'):
        print("saving model...")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for i, submodel in enumerate(self.submodels):
            save_path = os.path.join(save_dir, f'text_{i}_submodel.onnx')
            onnx.save(submodel, save_path)
        print("model saved in {} !".format(save_dir))