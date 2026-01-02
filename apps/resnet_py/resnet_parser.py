import pyds
import ctypes
import numpy as np

frame_idx = 1

def nvds_infer_parse_custom_resnet(tensor_meta):
    """
    
    """
    global frame_idx
    # Iterate through output layers
    # print("tensor_meta.num_output_layers", tensor_meta.num_output_layers)
    for i in range(tensor_meta.num_output_layers):
        layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
        layer_name = layer.layerName
        # print("layer_name", layer_name, layer.inferDims.numDims, )
        # print(layer.inferDims.numElements)
        # print(layer.inferDims.d)
        dims = layer.inferDims.d[:layer.inferDims.numDims] # Get dimensions
        # print(dims)
        # Access the raw tensor data
        # Convert this to a NumPy array for easier processing
        # The layer.buffer points to the raw memory
        ptr = pyds.get_ptr(layer.buffer)
        # Example conversion to numpy array (assuming float32 for output)
        # Be careful about the data type and dimensions based on your model's output
        # For example, if it's float32, use ctypes.c_float
        np_array = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=dims)
         
        # custom post-processing
        # For example, you might run custom NMS or other logic here.
        lean_array = np.squeeze(np_array)
        # print(f"frame {frame_idx} lean_array", lean_array[:10])
        frame_idx += 1
        predicted_class_index = np.argmax(lean_array, axis=0)
        print(f"frame {frame_idx} class {predicted_class_index}")

    output = "class_" + str(predicted_class_index)
    return output