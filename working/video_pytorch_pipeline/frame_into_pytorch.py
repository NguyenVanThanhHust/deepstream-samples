import os, sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import torch, torchvision
import cv2
import json

frame_format, pixel_bytes = 'RGBA', 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

CLASS_NAMES = detector.names
print(f'Loaded YOLOv5 model with {len(CLASS_NAMES)} classes: {CLASS_NAMES}')
ENABLE_VIDEO_OUTPUT = True

def create_video_writer(width=1920, height=1080, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter(
        "detections.mp4",
        fourcc,
        fps,
        (width, height)
    )

    return video_writer


if ENABLE_VIDEO_OUTPUT:
    video_writer = create_video_writer()
else:
    video_writer = None

frame_results = []
frame_id = 0

Gst.init(None)
pipeline = Gst.parse_launch(
    f'''
    filesrc location=/workspace/deepstream-samples/videos/scenario_1/scenario1_cam1_short.mkv num_buffers=400 !
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

def draw_detections(image, detections):

    h, w = image.shape[:2]

    sx = w / 640.0
    sy = h / 640.0

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        x1 = int(x1 * sx)
        x2 = int(x2 * sx)

        y1 = int(y1 * sy)
        y2 = int(y2 * sy)

        label = (
            f'{det["class_name"]} '
            f'{det["confidence"]:.2f}'
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return image

def on_frame_probe(pad, info):
    global frame_id
    global video_writer

    buf = info.get_buffer()
    print(f'[{buf.pts / Gst.SECOND:6.2f}]')

    im_tensor, rgb = buffer_to_tensor(
        buf, 
        pad.get_current_caps()
    )
    print("before infer")

    with torch.no_grad():
        pred_tensor = detector(im_tensor)

    print("pred_tensor.shape", pred_tensor.shape)

    print(
        "obj:",
        pred_tensor[...,4].min().item(),
        pred_tensor[...,4].max().item()
    )

    print(
        "cls:",
        pred_tensor[...,5:].min().item(),
        pred_tensor[...,5:].max().item()
    )

    print("after infer")
    print(type(pred_tensor))
    print(pred_tensor.shape if hasattr(pred_tensor, "shape") else "no shape")
    detections = parse_yolo_raw_output(
        pred_tensor,
        CLASS_NAMES
    )

    print(
        "confidence range:",
        (pred_tensor[...,4] * pred_tensor[...,5:].max(dim=-1).values).max().item()
    )
    print("num detections:", len(detections))

    print("num detections:", len(detections))

    if len(detections):
        print(detections[:5])

    frame_results.append({
        "frame_id": frame_id,
        "pts": float(buf.pts / Gst.SECOND),
        "detections": detections
    })

    if ENABLE_VIDEO_OUTPUT:
        vis = draw_detections(
            rgb.copy(),
            detections
        )
        cv2.imwrite(
            f'logs/frame_{frame_id:04d}.jpg',
            cv2.cvtColor(
                vis,
                cv2.COLOR_RGB2BGR
            )
        )
        video_writer.write(
            cv2.cvtColor(
                vis,
                cv2.COLOR_RGB2BGR
            )
        )

    frame_id += 1

    return Gst.PadProbeReturn.OK
def buffer_to_tensor(buf, caps):
    caps_structure = caps.get_structure(0)
    width = caps_structure.get_value('width')
    height = caps_structure.get_value('height')

    is_mapped, map_info = buf.map(Gst.MapFlags.READ)
    if not is_mapped:
        return None, None

    try:
        rgba = np.ndarray(
            shape=(height, width, pixel_bytes),
            dtype=np.uint8,
            buffer=map_info.data
        ).copy()

        rgb = rgba[:, :, :3]

        rgb_resized = cv2.resize(rgb, (640, 640))

        tensor = (
            torch.from_numpy(rgb_resized)
            .permute(2, 0, 1)
            .float()
            / 255.0
        ).unsqueeze(0).to(device)

        return tensor, rgb

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
    with open("detections.json", "w") as f:
        json.dump(
            frame_results,
            f,
            indent=2
        )

    open(f'logs/{os.path.splitext(sys.argv[0])[0]}.pipeline.dot', 'w').write(
        Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
    )
    pipeline.set_state(Gst.State.NULL)