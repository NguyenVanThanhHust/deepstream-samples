import os, sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import torch, torchvision

frame_format, pixel_bytes = 'RGBA', 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((640, 640)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = detector.names
print(f'Loaded YOLOv5 model with {len(CLASS_NAMES)} classes: {CLASS_NAMES}')

Gst.init(None)
pipeline = Gst.parse_launch(
    f'''
    filesrc location=/workspace/deepstream-samples/videos/scenario_1/scenario1_cam1_short.mkv  num-buffers=200 !
    decodebin !
    nvvideoconvert !
    video/x-raw, format={frame_format} !
    fakesink name=final_sink
'''
)

def parse_yolo_raw_output(
    pred,
    class_names=None,
    conf_thres=0.25,
    iou_thres=0.45
):
    """
    pred:
        torch.Tensor [1,25200,85]

    returns:
        list[dict]
    """

    pred = pred[0]

    boxes_xywh = pred[:, :4]
    objectness = pred[:, 4]

    class_scores = pred[:, 5:]

    class_conf, class_id = class_scores.max(dim=1)

    confidence = objectness * class_conf

    keep = confidence > conf_thres

    boxes_xywh = boxes_xywh[keep]
    confidence = confidence[keep]
    class_id = class_id[keep]

    if len(boxes_xywh) == 0:
        return []

    x = boxes_xywh[:, 0]
    y = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]

    boxes_xyxy = torch.stack([
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    ], dim=1)

    keep_idx = torchvision.ops.nms(
        boxes_xyxy,
        confidence,
        iou_thres
    )

    boxes_xyxy = boxes_xyxy[keep_idx]
    confidence = confidence[keep_idx]
    class_id = class_id[keep_idx]

    detections = []

    for box, conf, cls in zip(
        boxes_xyxy,
        confidence,
        class_id
    ):

        cls = int(cls)

        detections.append({
            "bbox": [
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3])
            ],
            "confidence": float(conf),
            "class_id": cls,
            "class_name":
                class_names[cls]
                if class_names is not None
                else str(cls)
        })

    return detections

def on_frame_probe(pad, info):
    buf = info.get_buffer()
    print(f'[{buf.pts / Gst.SECOND:6.2f}]')

    im_tensor = buffer_to_tensor(buf, pad.get_current_caps())
    with torch.no_grad():
        results = detector(im_tensor)
    print(results.shape) 
    return Gst.PadProbeReturn.OK

def buffer_to_tensor(buf, caps):
    casp_structure = caps.get_structure(0)
    width = casp_structure.get_value('width')
    height = casp_structure.get_value('height')

    is_mapped, map_info = buf.map(Gst.MapFlags.READ)
    if is_mapped:
        # Process the mapped buffer data
        try:
            im_array = np.ndarray(
                shape=(height, width, pixel_bytes), 
                dtype=np.uint8, 
                buffer=map_info.data
            ).copy()
            return preprocess(im_array[:,:,:3]).unsqueeze(0).to(device)
        finally:
            buf.unmap(map_info)


pipeline.get_by_name('final_sink').get_static_pad('sink').add_probe(
    Gst.PadProbeType.BUFFER,
    on_frame_probe
)

pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        msg = pipeline.get_bus().timed_pop_filtered(
            Gst.SECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        if msg:
            text = msg.get_structure().to_string() if msg.get_structure() else ''
            msg_type = Gst.message_type_get_name(msg.type)
            print(f'{msg.src.name}: [{msg_type}] {text}')
            break
finally:
    open(f'logs/{os.path.splitext(sys.argv[0])[0]}.pipeline.dot', 'w').write(
        Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
    )
    pipeline.set_state(Gst.State.NULL)