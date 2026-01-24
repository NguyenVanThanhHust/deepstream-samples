# Application
For Python apps, I would use nvinfer-server to use Triton server. This would keep everything in Python
For CPP apps, I would user nvinfer to use TensorRT.
## For learning
You could learn in following order.
* [im_cls_py](im_cls_py) -- very first deepstream app with Resnet18 model. This is the first pipeline that check if the deepstream is running. In this sample, we will parse output tensor. This is corresponding to [deepstream-test1](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-test1). There is [Cpp version](im_cls_cpp) too.

* [yolov1_py](yolov1_py) -- This app is developed from above but replace image classification to object detection. We use Yolov1 for simple demonstration for simple object detection output detection parser. There is [cpp version](yolov1_cpp) too.

* [yolov3_py](yolov3_py) -- This app is developed from above but replace yolov1  to yolov3. This is to demonstrate more complex output parser, from output head to multiple output head. There is [cpp version](yolov3_cpp) too.

* [yolov3_segmen_py](yolov3_segmen_py) -- This app is developed from above but replace yolov3  to yolov3 segmentation. This is to demonstrate segmentation output parser. There is [cpp version](yolov3_segmen_cpp) too.

* [yolov3_pose_estimation_py](yolov3_pose_estimation_py) -- This app is developed from above but replace yolov3  to yolov3 pose estimation. This is to demonstrate pose estimation output parser. There is [cpp version](yolov3_pose_estimation_cpp) too.

* [multi_source_reading_py](multi_source_reading_py) -- we replace previous `h264 parser` with `decodebin`. This will automatically pick whatever decoder that available for to decode RTSP stream or video input. There is [cpp version](multi_source_reading_cpp) too.

* [different_type_outputs](different_type_outputs) -- we show diffrenet type of outputs if deepstream.

## For application
After finishing learning section, you could use below for more performance application

* [rtdetr_py](rtdetr_py) -- This app is developed from yolov3 object detection but replace yolov3 to RTDETR. In my opinion, this is the best compromise for CCTV application. There is [cpp version](rtdetr_cpp) too.

* [gdino_py](gdino_py) -- This app is developed from yolov3 object segmentation but replace yolov3 to Grounding DINO. In my opinion, this is the best object detection and object segmentation. I'm not sure what is its application. There is [cpp version](gdino_cpp) too.
