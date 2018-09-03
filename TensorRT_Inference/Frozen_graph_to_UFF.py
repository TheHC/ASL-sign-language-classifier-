import os
import uff


FROZEN_GRAPH_FILENAME='../Model/model0/frozen_model.pb'
OUTPUT_NAME='logits/BiasAdd'
OUTPUT_UFF_FILE_NAME='../TensorRT_Inference/UFF.uff'

def UFF_write():
	uff.from_tensorflow_frozen_model(
		frozen_file=FROZEN_GRAPH_FILENAME,
		output_nodes=[OUTPUT_NAME],
		output_filename=OUTPUT_UFF_FILE_NAME,
		text=False,
		)
if __name__ == '__main__' : 
	UFF_write()
