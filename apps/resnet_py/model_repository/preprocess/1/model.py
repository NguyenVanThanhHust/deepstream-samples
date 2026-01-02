import numpy as np
import json

import triton_python_backend_utils as pb_utils

from PIL import Image
import numpy as np
import cv2
from datetime import datetime

def preprocess_opencv(image_array: np.ndarray) -> np.ndarray:
    """
    Args:
        image_array: The input image as a NumPy array (H x W x C, typically uint8).

    Returns:
        The processed image as a NumPy array (C x H x W, typically float32),
        ready for model input.
    """
    # 1. Convert BGR → RGB
    # img_rgb = cv2.cvtColor(np.moveaxis(image_array, 0, -1), cv2.COLOR_BGR2RGB)
    # because output of nv decoder is rgb format, no need to convert from BGR2RGB
    img_rgb = np.moveaxis(image_array, 0, -1)

    # 2. Convert to PIL (so resizing is identical to PyTorch)
    pil_img = Image.fromarray(img_rgb)

    # 3. PIL Resize (bilinear) – same as PyTorch
    pil_img = pil_img.resize((224, 224), Image.BILINEAR)

    # 4. PIL → NumPy (float32, scaled 0–1)
    img = np.array(pil_img).astype(np.float32) / 255.0  # HWC

    # 5. Normalize (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std

    # 6. HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    return img


class TritonPythonModel:

    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "PREPROCESS_OUTPUT_0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        
    def execute(self, requests):
        output0_dtype = self.output0_dtype
        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "PREPROCESS_INPUT_0")
            img_uint8 = in_0.as_numpy()

            # Undo the expand dims
            img_uint8 = np.squeeze(img_uint8, axis=0)
            preprocessed_image = preprocess_opencv(img_uint8)
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

            img_fp32 = np.ascontiguousarray(preprocessed_image, dtype='float32')
 
            out_tensor_0 = pb_utils.Tensor("PREPROCESS_OUTPUT_0",
                                           img_fp32.astype(output0_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up preprocessing...')