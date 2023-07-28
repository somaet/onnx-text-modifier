import os
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

class onnxText: 
    def __init__(self, content_list):
        self.nodes = content_list
        
    @classmethod 
    def from_stream(cls, stream):
        print("load text...")
        stream.seek(0)
        content_list = [line.decode('utf-8').strip() for line in stream]
        stream.close()
        return cls(content_list)
    


class onnxDownload: 
    def __init__(self, model_proto, subgraph, submodels):
        self.model_proto = model_proto
        self.subgraph = subgraph
        self.submodels = submodels
        

    @classmethod 
    def from_model(cls, model_proto, subgraph):
        # Define the subgraphs
        print("hello")
        subgraphs = [
            {
                "inputs": [subgraph.nodes[0], subgraph.nodes[1]],
                "outputs": [subgraph.nodes[2], subgraph.nodes[3]]
            },
            {
                "inputs": [subgraph.nodes[4]],
                "outputs": [subgraph.nodes[5]]
            }
        ]
        print("hello")

        # Extract subgraphs
        submodels = []
        for subgraph in subgraphs:
            nodes = []
            inputs = [helper.make_tensor_value_info(i, TensorProto.FLOAT, [1, 3, 224, 224]) for i in subgraph["inputs"]]
            outputs = [helper.make_tensor_value_info(o, TensorProto.FLOAT, [1, 1000]) for o in subgraph["outputs"]]
            for node in model_proto.graph.node:
                if node.name in subgraph["inputs"] or node.name in subgraph["outputs"]:
                    nodes.append(node)
            subgraph = helper.make_graph(nodes, "subgraph", inputs, outputs)
            submodel = helper.make_model(subgraph, producer_name='onnx-submodels')
            submodels.append(submodel)
            
        return cls(model_proto, subgraph, submodels)
    
    def save_model(self, save_dir='./text_onnx'):
        print("saving model...")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for i, submodel in enumerate(self.submodels):
            save_path = os.path.join(save_dir, f'text_{i}_submodel.onnx')
            onnx.save(submodel, save_path)
        print("model saved in {} !".format(save_dir))