import time, os, sys, logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Lambda,
)

try:
    import xir
    import vart
except:
    pass

def runDPU_(x, img):
    inputTensors = x.get_input_tensors()
    outputTensors = x.get_output_tensors()
    
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = []
    for i in range(len(outputTensors)):
        output_ndim.append(tuple(outputTensors[i].dims))
    
    outputs = [] # Reset output data, if not it will segment fault when run DPU.
    inputData = []
    inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]

    for i in range(len(outputTensors)):
        outputs.append(np.empty(output_ndim[i], dtype=np.float32, order="C"))

    '''init input image to input buffer '''
    imageRun = inputData[0]
    imageRun[0, ...] = img[0].reshape(input_ndim[1:])

    '''init input image to input buffer '''
    '''run with batch '''

    time_pred_start = time.time()
    x.predict(inputData, outputs)
    logging.debug("Pred times (DPU function) = {:.4f} seconds".format(time.time() - time_pred_start))
    return outputs

class XMODEL:
    def __init__(self, model, name):
        self.__m = model # model path
        self.subgraphs = list()
        self.runner = None
        self.name = name
        self.outputs = []

    def get_child_subgraph_dpu(self) -> list:
        graph = xir.Graph.deserialize(self.__m)
        assert graph is not None, "'graph' should not be None."

        root_subgraph = graph.get_root_subgraph()
        assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."

        if root_subgraph.is_leaf:
            return []

        child_subgraphs = root_subgraph.toposort_child_subgraph()
        assert child_subgraphs is not None and len(child_subgraphs) > 0

        self.subgraphs = [
            cs
            for cs in child_subgraphs
            if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
        ]

        return self.subgraphs
    
    def get_dpu_runner(self, graph=None):
        if graph is None:
            self.runner = vart.Runner.create_runner(self.subgraphs[0], 'run')
        else:
            self.runner = vart.Runner.create_runner(graph[0], 'run')
        return self.runner
    
    def init(self):
        self.get_child_subgraph_dpu()
        self.get_dpu_runner()

    def predict(self, input_data, outputs):
        job_id = self.runner.execute_async(input_data, outputs)
        self.runner.wait(job_id)

        return outputs
    
    def get_input_tensors(self):
        if self.runner is None:
            return None
        else:
            return self.runner.get_input_tensors()
    
    def get_output_tensors(self):
        if self.runner is None:
            return None
        else:
            return self.runner.get_output_tensors()


class IVIT_MODEL(object):
    
    def __init__(self):
        pass
    
    def get_dets_pattern(self):

        temp_dets = {                      
            "id": None, 
            "label": None,
            "score": None,
            "xmin": None,
            "ymin": None,
            "xmax": None,
            "ymax": None,
        }

        return temp_dets