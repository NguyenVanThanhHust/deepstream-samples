import torch
import torchvision.models as models
import onnx

def convert_resnet18_to_onnx(output_path="resnet18.onnx"):
    # 1. Load the pre-trained ResNet18 model
    # Set 'weights' to 'ResNet18_Weights.DEFAULT' to use the latest pretrained weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # It's crucial to set the model to evaluation mode for export
    # This disables layers like dropout and sets batchnorm layers correctly
    model.eval()

    # 2. Define a dummy input
    # ResNet18 expects an input of shape (batch_size, channels, height, width)
    # For ImageNet, this is typically (1, 3, 224, 224)
    # Using 'torch.randn' creates a tensor with random values of the correct size
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

    # 3. Export the model to ONNX format
    torch.onnx.export(model,  # The PyTorch model
                      dummy_input,  # A dummy input to trace the model's computation graph
                      output_path,  # Where to save the ONNX file
                      export_params=True,  # Export model parameters (weights)
                      opset_version=11,  # ONNX opset version (recommend 11 or higher)
                      do_constant_folding=True,  # Optimize constant values
                      input_names=['input'],  # Name the input node
                      output_names=['output'],  # Name the output node
                      dynamic_axes={'input' : {0 : 'batch_size'},    # Specify dynamic batch size
                                    'output' : {0 : 'batch_size'}})

    print(f"Model successfully converted to {output_path}")

    # Optional: Verify the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model checker passed!")

if __name__ == "__main__":
    convert_resnet18_to_onnx()
